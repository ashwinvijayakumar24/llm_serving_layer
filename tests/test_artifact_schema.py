"""
Artifact schema and provenance tests. Pure CPU, no GPU, runs in CI.

Each test corresponds to a risk in docs/RISK_REGISTER.md that would otherwise
produce a plausible wrong number with no error anywhere.

    pytest tests/test_artifact_schema.py -v
"""

import json

import pytest

from serving.metrics.artifact import (
    PUBLISHABLE_QOS,
    REGISTRY,
    Artifact,
    MetricRegistry,
    MetricSpec,
    Provenance,
    assert_comparable,
)


def make_provenance(**overrides):
    base = dict(
        timestamp_utc="2026-07-31T20:00:00+00:00",
        hostname="atl1-1-02-018-33-0.pace.gatech.edu",
        platform="Linux-5.14",
        slurm_job_id="11577991",
        slurm_node="atl1-1-02-018-33-0",
        slurm_partition="gpu-l40s",
        slurm_qos="inferno",
        repo_sha="deadbeef",
        repo_dirty=False,
        engine_tag="v0.1.0",
        engine_sha="6ff40a1",
        seed=1234,
    )
    base.update(overrides)
    return Provenance(**base)


def make_artifact(**overrides):
    a = Artifact(
        name="goodput_sweep",
        provenance=overrides.pop("provenance", make_provenance()),
        config={"arrival": "poisson", "rate_rps": 8.0},
        window={"start_s": 30.0, "end_s": 330.0},
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


# --------------------------------------------------------------------------
# R16 — metric name collisions
# --------------------------------------------------------------------------


def test_registry_rejects_name_collision():
    """
    The engine's live instance of this bug: `peak_mem_mb` means host RSS in
    bench/harness.py:44-47 and CUDA max_memory_allocated in
    bench/baseline_hf.py:108 — same column, different quantity, silently
    compared (BENCHMARKS.md:247).
    """
    reg = MetricRegistry()
    reg.register(MetricSpec("peak_mem_mb", "memory", "MB", "host RSS"))
    with pytest.raises(ValueError, match="Metric name collision"):
        reg.register(MetricSpec("peak_mem_mb", "memory", "MB", "torch.cuda.max_memory_allocated"))


def test_registry_allows_identical_reregistration():
    """Re-registering the same semantics is a no-op, not an error."""
    reg = MetricRegistry()
    spec = MetricSpec("ttft_ms", "time to first token", "ms", "client wall clock")
    reg.register(spec)
    reg.register(MetricSpec("ttft_ms", "time to first token", "ms", "client wall clock"))
    assert reg.get("ttft_ms").source == "client wall clock"


def test_gpu_and_host_memory_are_distinct_metrics():
    """The fix for the collision: distinct names, distinct sources."""
    assert REGISTRY.get("gpu_mem_mb").source != REGISTRY.get("host_rss_mb").source
    assert REGISTRY.get("gpu_mem_mb").unit == REGISTRY.get("host_rss_mb").unit == "MB"


def test_unregistered_metric_rejected():
    a = make_artifact()
    with pytest.raises(KeyError, match="Unregistered metric"):
        a.add_samples("throughput", [1.0])       # ambiguous name, deliberately absent


# --------------------------------------------------------------------------
# R13 — embers QOS must not back a published number
# --------------------------------------------------------------------------


def test_embers_qos_is_not_publishable():
    """
    `embers` is free and preemptible. A preempted run truncates its measurement
    window into something that looks exactly like a completed short run.
    """
    ok, reasons = make_provenance(slurm_qos="embers").is_publishable()
    assert not ok
    assert any("embers" in r or "preempt" in r.lower() for r in reasons)


def test_inferno_qos_is_publishable():
    ok, reasons = make_provenance().is_publishable()
    assert ok, reasons
    assert "inferno" in PUBLISHABLE_QOS


@pytest.mark.parametrize(
    "override,needle",
    [
        ({"repo_dirty": True}, "dirty"),
        ({"repo_sha": None}, "SHA"),
        ({"seed": None}, "seed"),
        ({"engine_tag": None, "engine_sha": None}, "engine"),
    ],
)
def test_unpublishable_conditions(override, needle):
    ok, reasons = make_provenance(**override).is_publishable()
    assert not ok
    assert any(needle.lower() in r.lower() for r in reasons)


def test_unpublishable_artifact_is_still_written(tmp_path):
    """
    A rejected artifact is real data and must not be discarded — only barred from
    backing a claim. Losing the run would be worse than keeping it labelled.
    """
    a = make_artifact(provenance=make_provenance(slurm_qos="embers"))
    a.add_samples("ttft_ms", [12.0, 15.0])
    path = a.write(tmp_path)
    payload = json.loads(path.read_text())
    assert payload["publishable"] is False
    assert payload["publishable_blockers"]
    assert payload["samples"]["ttft_ms"] == [12.0, 15.0]


# --------------------------------------------------------------------------
# R12 — cross-allocation comparison
# --------------------------------------------------------------------------


def test_same_allocation_is_comparable():
    a = make_artifact()
    b = make_artifact()
    assert_comparable(a, b)          # must not raise


def test_cross_allocation_comparison_raises():
    """
    Not a warning. The engine's own throughput moved ~79 -> ~60 tok/s across
    nodes for identical code (BENCHMARKS.md:17), so a cross-allocation delta
    measures node assignment. A warning in a notebook is a warning nobody reads.
    """
    a = make_artifact()
    b = make_artifact(provenance=make_provenance(slurm_job_id="99999999", slurm_node="other-node"))
    with pytest.raises(ValueError, match="Refusing to compare across allocations"):
        assert_comparable(a, b)


def test_missing_allocation_id_blocks_comparison():
    """A run outside Slurm has no allocation identity, so comparability is unverifiable."""
    a = make_artifact(provenance=make_provenance(slurm_job_id=None))
    b = make_artifact()
    with pytest.raises(ValueError, match="allocation id missing"):
        assert_comparable(a, b)


# --------------------------------------------------------------------------
# R11 / R15 — measurement window and raw samples
# --------------------------------------------------------------------------


def test_samples_without_window_rejected():
    """
    Including the drain tail flatters p99 as the queue empties, and the run looks
    entirely normal. Explicit window boundaries are mandatory.
    """
    a = Artifact(name="no_window", provenance=make_provenance())
    a.add_samples("itl_ms", [10.0, 11.0])
    assert any("window" in p.lower() for p in a.validate())
    with pytest.raises(ValueError, match="not well-formed"):
        a.write("/tmp/should-not-be-written")


def test_raw_samples_are_preserved_not_summarized(tmp_path):
    """
    Percentiles must be computed at analysis time over the pooled request set.
    The engine stores per-run p50/p99 and discards samples
    (bench/harness.py:136-137,178), which makes correct pooling impossible after
    the fact.
    """
    a = make_artifact()
    values = [float(i) for i in range(200)]
    a.add_samples("itl_ms", values)
    payload = json.loads(a.write(tmp_path).read_text())
    assert payload["samples"]["itl_ms"] == values, "raw samples must survive round-trip"


def test_metric_specs_travel_with_the_artifact(tmp_path):
    """A reader must be able to interpret a number without this source tree."""
    a = make_artifact()
    a.add_samples("ttft_ms", [20.0])
    a.set_scalar("goodput_rps", 7.5)
    payload = json.loads(a.write(tmp_path).read_text())
    specs = payload["metric_specs"]
    assert set(specs) == {"ttft_ms", "goodput_rps"}
    assert specs["ttft_ms"]["unit"] == "ms"
    assert "intended dispatch" in specs["ttft_ms"]["source"].lower()


def test_offered_load_and_throughput_are_distinct():
    """
    Offered load is a PARAMETER; throughput is an OUTCOME. Reporting one as the
    other is a classic serving-benchmark error, so they cannot share a name.
    """
    assert REGISTRY.get("offered_load_rps").quantity != REGISTRY.get("goodput_rps").quantity
    assert "configuration" in REGISTRY.get("offered_load_rps").source


def test_ttft_measured_from_intended_dispatch():
    """R1, coordinated omission — encoded in the metric definition itself."""
    assert "intended dispatch" in REGISTRY.get("ttft_ms").source.lower()
    assert "dispatch_drift_ms" in REGISTRY.all()


# --------------------------------------------------------------------------
# Provenance capture
# --------------------------------------------------------------------------


def test_capture_records_environment_without_fabricating(tmp_path):
    """
    Absent provenance is recorded as None, never guessed. tmp_path is not a git
    repo, so SHA fields must come back None rather than inheriting the cwd's.
    """
    p = Provenance.capture(repo_root=tmp_path, seed=42)
    assert p.seed == 42
    assert p.hostname and p.timestamp_utc and p.python_version
    assert p.repo_sha is None
    assert p.engine_tag is None and p.engine_sha is None


def test_allocation_id_shape():
    p = make_provenance()
    assert p.allocation_id == "11577991@atl1-1-02-018-33-0"
    assert make_provenance(slurm_job_id=None).allocation_id is None
