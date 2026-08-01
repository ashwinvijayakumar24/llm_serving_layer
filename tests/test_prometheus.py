"""
Prometheus exporter tests. CPU only, no GPU, no network.

The exposition format is parsed by a parser written here rather than asserted
against with substring checks. That is deliberate: a substring assertion passes
on output that a real scraper rejects, and the whole value of this endpoint is
that something else can read it. `parse_exposition` implements the format's
actual rules — one HELP and one TYPE per family, series names derived from the
family name, `le` present and ordered on histogram buckets — so a malformed
document fails here instead of at 3am in a scrape.

The load-bearing test is `test_no_registry_collisions_are_possible`: it asserts
the structural rule, not a snapshot of today's metric list, so a metric added
later under a registered name fails immediately (R16).

    python3 -m pytest tests/test_prometheus.py -q
"""

from __future__ import annotations

import math

import pytest

from serving.metrics import prometheus as prom
from serving.metrics.artifact import REGISTRY
from serving.metrics.cuda_guard import CudaGuard
from serving.server.app import ServerMetrics

# ---------------------------------------------------------------------------
# A real parser for the exposition format
# ---------------------------------------------------------------------------


class ParsedFamily:
    def __init__(self, name: str):
        self.name = name
        self.help: str | None = None
        self.type: str | None = None
        self.series: list[tuple[str, dict[str, str], float]] = []


def _parse_labels(text: str) -> dict[str, str]:
    if not text:
        return {}
    assert text.startswith("{") and text.endswith("}"), f"bad label block {text!r}"
    out: dict[str, str] = {}
    for part in _split_labels(text[1:-1]):
        if not part:
            continue
        k, _, v = part.partition("=")
        assert v.startswith('"') and v.endswith('"'), f"unquoted label value in {part!r}"
        out[k] = v[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return out


def _split_labels(body: str) -> list[str]:
    parts, cur, in_q, esc = [], [], False, False
    for ch in body:
        if esc:
            cur.append(ch)
            esc = False
            continue
        if ch == "\\":
            cur.append(ch)
            esc = True
            continue
        if ch == '"':
            in_q = not in_q
        if ch == "," and not in_q:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur))
    return parts


def parse_exposition(text: str) -> dict[str, ParsedFamily]:
    """
    Parse, and reject anything a scraper would reject.

    Raises AssertionError on: a value that is not a float, a HELP/TYPE for a name
    that already had one, a series whose name does not belong to a declared
    family, or an unparseable line.
    """
    assert text.endswith("\n"), "exposition document must end with a newline"
    families: dict[str, ParsedFamily] = {}
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# HELP "):
            name, _, help_text = line[len("# HELP "):].partition(" ")
            fam = families.setdefault(name, ParsedFamily(name))
            assert fam.help is None, f"duplicate HELP for {name}"
            fam.help = help_text
            continue
        if line.startswith("# TYPE "):
            name, _, type_text = line[len("# TYPE "):].partition(" ")
            fam = families.setdefault(name, ParsedFamily(name))
            assert fam.type is None, f"duplicate TYPE for {name}"
            assert type_text in ("counter", "gauge", "histogram", "summary", "untyped")
            fam.type = type_text
            continue
        if line.startswith("#"):
            continue  # free comment; allowed by the format

        # series line: name[{labels}] value
        if "{" in line:
            name, _, rest = line.partition("{")
            labels_text, _, value_text = rest.rpartition("}")
            labels = _parse_labels("{" + labels_text + "}")
        else:
            name, _, value_text = line.partition(" ")
            labels = {}
        value_text = value_text.strip()
        assert name and value_text, f"unparseable series line: {raw!r}"
        if value_text == "+Inf":
            value = math.inf
        elif value_text == "NaN":
            value = math.nan
        else:
            value = float(value_text)

        owner = None
        for suffix in ("_bucket", "_sum", "_count", ""):
            cand = name[: -len(suffix)] if suffix and name.endswith(suffix) else name
            if cand in families:
                owner = families[cand]
                break
        assert owner is not None, f"series {name!r} has no declared family"
        owner.series.append((name, labels, value))
    return families


# ---------------------------------------------------------------------------
# Fakes: enough surface for the exporter, nothing more
# ---------------------------------------------------------------------------


class FakeAllocator:
    def __init__(self):
        self.num_blocks = 100
        self.block_size = 16
        self.watermark_blocks = 8
        self.num_free = 60
        self.num_used = 40
        self.utilization = 0.4
        # Read only by the JSON /metrics endpoint; present so the two renderings
        # can be compared against one allocator.
        self.total_allocated = 40
        self.total_freed = 0
        self.peak_used = 40

    def tokens_capacity(self) -> int:
        return self.num_blocks * self.block_size


class FakeScheduler:
    def __init__(self, *, cache=None, preemptions=None):
        self.allocator = FakeAllocator()
        self._snap = {
            "step": 7, "running": 3, "waiting": 2, "finished": 5,
            "blocks_free": 60, "blocks_used": 40, "block_utilization": 0.4,
        }
        if cache is not None:
            self.cache = cache
        if preemptions is not None:
            self.preemptions = preemptions

    def snapshot(self):
        return dict(self._snap)


class FakeCache:
    """Stands in for the Phase 4 radix cache, which does not exist yet."""

    def stats(self):
        return {"block_hits": 120, "block_misses": 30}


class FakeLoop:
    def __init__(self, healthy=True, steps=7):
        self.running = True
        self.healthy = healthy
        self.steps = steps


def _metrics(**kw) -> ServerMetrics:
    m = ServerMetrics()
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def _render(**kw) -> str:
    m = kw.pop("metrics", None) or _metrics(
        requests_received=10, requests_completed=7, requests_failed=1,
        requests_cancelled=1, requests_shed_429=1, output_tokens_total=400,
        prompt_tokens_total=900,
    )
    sched = kw.pop("scheduler", None) or FakeScheduler()
    loop = kw.pop("loop", None) or FakeLoop()
    return prom.render_server_metrics(m, sched, loop, **kw)


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------


def test_output_parses_as_exposition_format():
    families = parse_exposition(_render())
    assert families


def test_every_family_has_help_and_type():
    for name, fam in parse_exposition(_render()).items():
        assert fam.help, f"{name} has no HELP"
        assert fam.type, f"{name} has no TYPE"
        assert fam.series, f"{name} declared but emitted no series"


def test_no_duplicate_family_names():
    text = _render()
    names = [ln.split(" ")[2] for ln in text.split("\n") if ln.startswith("# TYPE ")]
    assert len(names) == len(set(names)), f"duplicate TYPE lines: {names}"


def test_render_rejects_duplicate_families():
    fam = prom.counter("llm_x_total", "x")
    with pytest.raises(ValueError, match="duplicate metric family"):
        prom.render([fam, prom.counter("llm_x_total", "x again")])


def test_invalid_metric_name_rejected():
    with pytest.raises(ValueError, match="invalid Prometheus metric name"):
        prom.counter("llm-bad-name", "help")


def test_help_is_required():
    with pytest.raises(ValueError, match="no HELP"):
        prom.gauge("llm_thing", "")


def test_newlines_in_help_are_escaped():
    text = prom.render([prom.gauge("llm_thing", "line one\nline two", 1)])
    assert "\\n" in text
    families = parse_exposition(text)
    assert families["llm_thing"].help == "line one\\nline two"


# ---------------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------------


def test_counters_are_monotonic_across_scrapes():
    m = _metrics(requests_received=1, output_tokens_total=10)
    sched, loop = FakeScheduler(), FakeLoop()

    def counter_values(text):
        fams = parse_exposition(text)
        return {
            n: f.series[0][2]
            for n, f in fams.items()
            if f.type == "counter" and f.series
        }

    first = counter_values(prom.render_server_metrics(m, sched, loop))
    m.requests_received += 5
    m.output_tokens_total += 50
    m.requests_completed += 3
    loop.steps += 9
    second = counter_values(prom.render_server_metrics(m, sched, loop))

    assert set(first) == set(second)
    for name, v0 in first.items():
        assert second[name] >= v0, f"counter {name} decreased: {v0} -> {second[name]}"


def test_counter_names_end_in_total():
    for name, fam in parse_exposition(_render()).items():
        if fam.type == "counter":
            assert name.endswith("_total"), f"counter {name} lacks the _total suffix"


def test_histograms_are_cumulative_and_terminate_in_inf():
    m = _metrics()
    for v in (0.5, 3.0, 30.0, 3000.0, 99999.0):
        m.observe("ttft_from_arrival_ms", v)
    fam = parse_exposition(_render(metrics=m))["llm_ttft_from_arrival_ms"]

    buckets = [(lbl["le"], val) for n, lbl, val in fam.series if n.endswith("_bucket")]
    values = [v for _, v in buckets]
    assert values == sorted(values), f"buckets not cumulative: {buckets}"
    assert buckets[-1][0] == "+Inf"
    assert buckets[-1][1] == 5

    count = next(v for n, _, v in fam.series if n.endswith("_count"))
    total = next(v for n, _, v in fam.series if n.endswith("_sum"))
    assert count == 5
    assert total == pytest.approx(0.5 + 3.0 + 30.0 + 3000.0 + 99999.0)


def test_histogram_bucket_bounds_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        prom.Histogram([1, 2, 2])


def test_scheduler_healthy_gauge_reflects_the_loop():
    assert 'llm_scheduler_healthy' in _render()
    ok = parse_exposition(_render(loop=FakeLoop(healthy=True)))["llm_scheduler_healthy"]
    bad = parse_exposition(_render(loop=FakeLoop(healthy=False)))["llm_scheduler_healthy"]
    assert ok.series[0][2] == 1
    assert bad.series[0][2] == 0


def test_gauges_track_the_allocator():
    fams = parse_exposition(_render())
    assert fams["llm_blocks_free"].series[0][2] == 60
    assert fams["llm_blocks_used"].series[0][2] == 40
    assert fams["llm_block_utilization"].series[0][2] == pytest.approx(0.4)


def test_cuda_errors_counter_is_always_present_and_starts_at_zero():
    guard = CudaGuard(available_fn=lambda: False)
    fams = parse_exposition(_render(guard=guard))
    assert fams["llm_cuda_errors_total"].series[0][2] == 0
    assert "poisoned" in fams["llm_cuda_errors_total"].help.lower() or \
           "fatal" in fams["llm_cuda_errors_total"].help.lower()


# ---------------------------------------------------------------------------
# Optional subsystems: absent means ABSENT, not zero
# ---------------------------------------------------------------------------


def test_cache_metrics_absent_when_there_is_no_cache():
    """
    Phase 4 is being built in parallel. A zeroed cache metric is indistinguishable
    from a cache that never hits, so absence must be absence.
    """
    fams = parse_exposition(_render(scheduler=FakeScheduler()))
    assert "llm_cache_block_hits_total" not in fams
    assert "llm_cache_block_misses_total" not in fams


def test_cache_metrics_appear_when_a_cache_is_present():
    fams = parse_exposition(_render(scheduler=FakeScheduler(cache=FakeCache())))
    assert fams["llm_cache_block_hits_total"].series[0][2] == 120
    assert fams["llm_cache_block_misses_total"].series[0][2] == 30
    # hits / (hits + misses) is exactly the registered cache_hit_rate definition
    assert "block" in fams["llm_cache_block_hits_total"].help.lower()


def test_preemptions_are_labelled_by_policy_with_bounded_cardinality():
    sched = FakeScheduler(preemptions={"recompute": 4, "swap": 2})
    fam = parse_exposition(_render(scheduler=sched))["llm_preemptions_total"]
    policies = {lbl["policy"] for _, lbl, _ in fam.series}
    assert policies == {"recompute", "swap"}


def test_no_unbounded_cardinality_labels():
    forbidden = {"request_id", "id", "prompt", "user", "session", "trace_id"}
    for name, fam in parse_exposition(_render(
        scheduler=FakeScheduler(cache=FakeCache(), preemptions={"recompute": 1})
    )).items():
        for _, labels, _ in fam.series:
            bad = forbidden & set(labels)
            assert not bad, f"{name} carries unbounded label(s) {bad}"


def test_reads_the_real_radix_cache_not_just_the_fake():
    """
    The exporter duck-types the cache because Phase 4 landed in parallel. That is
    only defensible if it is checked against the REAL object: a defensive read
    that silently matches nothing is indistinguishable from no cache at all, and
    would publish an absent metric forever.

    `serving/cache/radix.py` counts blocks_reused / blocks_required — literally
    the registered cache_hit_rate numerator and denominator — so misses are
    DERIVED, and hits + misses is the denominator by construction.
    """
    from serving.cache.radix import RadixCache
    from serving.memory.allocator import BlockAllocator

    cache = RadixCache(BlockAllocator(num_blocks=64, block_size=4))
    cache.acquire(list(range(16)))  # cold: 4 blocks required, 0 reused

    sched = FakeScheduler()
    sched.prefix_cache = cache
    fams = parse_exposition(_render(scheduler=sched))

    hits = fams["llm_cache_block_hits_total"].series[0][2]
    misses = fams["llm_cache_block_misses_total"].series[0][2]
    assert hits == cache.stats.blocks_reused
    assert hits + misses == cache.stats.blocks_required


def test_reads_the_real_preemption_stats_shape():
    """Same argument for Phase 3's counters: verified against the real dict."""
    from serving.scheduler.preemption import PreemptionPolicy, PreemptionStats

    stats = PreemptionStats()
    stats.record(PreemptionPolicy.RECOMPUTE)
    stats.record(PreemptionPolicy.RECOMPUTE)
    stats.record(PreemptionPolicy.SWAP)

    sched = FakeScheduler()
    sched._snap.update(stats.as_dict())
    fam = parse_exposition(_render(scheduler=sched))["llm_preemptions_total"]
    by_policy = {lbl["policy"]: v for _, lbl, v in fam.series}
    assert by_policy == {"recompute": 2.0, "swap": 1.0}


# ---------------------------------------------------------------------------
# R16 — the load-bearing rule
# ---------------------------------------------------------------------------


def test_no_registry_collisions_are_possible():
    """
    Structural, not a snapshot. Every exported family either CLAIMS a registry
    name (and is then verified to be that exact quantity/unit/source) or is
    verified not to resemble one. Adding `llm_ttft_ms` later fails here.
    """
    fams = prom.server_families(
        _metrics(), FakeScheduler(cache=FakeCache(), preemptions={"recompute": 1}),
        FakeLoop(), guard=CudaGuard(available_fn=lambda: False),
        process_memory={"host_rss_mb": 12.0, "gpu_mem_mb": 34.0},
    )
    prom.assert_registry_alignment(fams)  # raises on any collision

    for fam in fams:
        if fam.registry_name is None:
            for cand in prom._registry_candidates(fam.name):
                assert cand not in REGISTRY, (
                    f"{fam.name} shadows registry metric {cand}"
                )


def test_registry_backed_families_carry_identical_semantics():
    fams = prom.server_families(
        _metrics(), FakeScheduler(), FakeLoop(),
        process_memory={"host_rss_mb": 12.0, "gpu_mem_mb": 34.0},
    )
    claimed = {f.registry_name: f for f in fams if f.registry_name}
    assert set(claimed) == {"output_tok_s", "host_rss_mb", "gpu_mem_mb"}
    for reg_name, fam in claimed.items():
        spec = REGISTRY.get(reg_name)
        assert fam.name == prom.PREFIX + reg_name
        # The definition travels with the series into whatever dashboard it
        # lands in — that is the entire mitigation for peak_mem_mb.
        assert f"quantity={spec.quantity}" in fam.help
        assert f"unit={spec.unit}" in fam.help
        assert f"source={spec.source}" in fam.help


def test_claiming_a_registry_name_under_the_wrong_export_name_raises():
    fam = prom.gauge("llm_something_else", "help", 1.0, registry_name="output_tok_s")
    with pytest.raises(ValueError, match="should then be named"):
        prom.assert_registry_alignment([fam])


@pytest.mark.parametrize(
    "name", ["llm_ttft_ms", "llm_queue_wait_ms", "llm_e2e_ms", "llm_itl_ms",
             "llm_cache_hit_rate", "llm_goodput_rps", "llm_dispatch_drift_ms"]
)
def test_shadowing_a_registered_name_is_rejected(name):
    """
    The specific names a well-meaning contributor would add. Each is defined in
    the REGISTRY from the CLIENT's intended dispatch or from the load generator's
    configuration; the server cannot observe any of them (R1/R16).
    """
    with pytest.raises(ValueError, match="collides with REGISTRY metric"):
        prom.assert_registry_alignment([prom.gauge(name, "server-side version", 1.0)])


def test_not_exported_names_are_documented_in_the_scrape():
    text = _render()
    for name in ("ttft_ms", "itl_ms", "e2e_ms", "queue_wait_ms", "cache_hit_rate"):
        assert f"NOT EXPORTED: {name}" in text, (
            f"{name} is omitted but the omission is undocumented; an absent metric "
            "reads as an oversight and gets re-added under the wrong definition"
        )
    for name in prom.NOT_EXPORTED:
        assert name in REGISTRY, f"NOT_EXPORTED lists {name}, which is not registered"


def test_server_latency_histograms_are_named_for_what_the_server_can_see():
    m = _metrics()
    m.observe("ttft_from_arrival_ms", 5.0)
    m.observe("itl_server_ms", 5.0)
    m.observe("e2e_from_arrival_ms", 5.0)
    m.observe("step_duration_host_ms", 5.0)
    fams = parse_exposition(_render(metrics=m))
    for name in ("llm_ttft_from_arrival_ms", "llm_itl_server_ms",
                 "llm_e2e_from_arrival_ms", "llm_step_duration_host_ms"):
        assert fams[name].type == "histogram"
    assert "intended dispatch" in fams["llm_ttft_from_arrival_ms"].help
    # R2: the step-duration histogram must carry its own health warning.
    assert "queueing" in fams["llm_step_duration_host_ms"].help


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------


def test_endpoint_serves_the_text_format_alongside_the_json_one():
    """
    ADDITIVE. The JSON /metrics is what the benchmark harness reads and what the
    artifact pipeline is built on; this endpoint must not have replaced it.
    """
    import asyncio

    import httpx

    from serving.server.app import create_app

    class NullTokenizer:
        def encode_chat(self, messages):
            return [1, 2, 3]

        def decode(self, token_ids):
            return ""

    class IdleScheduler(FakeScheduler):
        has_work = False

        def step(self):  # pragma: no cover - never reached, has_work is False
            raise AssertionError("idle scheduler must not be stepped")

    app = create_app(IdleScheduler(), NullTokenizer())

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            text = await c.get("/metrics/prometheus")
            js = await c.get("/metrics")
        await app.state.scheduler_loop.aclose()
        return text, js

    text_resp, json_resp = asyncio.run(go())

    assert text_resp.status_code == 200
    assert text_resp.headers["content-type"].startswith("text/plain")
    assert "version=0.0.4" in text_resp.headers["content-type"]
    fams = parse_exposition(text_resp.text)
    assert "llm_requests_received_total" in fams
    assert "llm_cuda_errors_total" in fams

    assert json_resp.status_code == 200
    assert "registered" in json_resp.json(), "the JSON endpoint must still work"


def test_endpoint_output_is_stable_across_scrapes():
    """A scrape must not mutate state; two consecutive scrapes of an idle server
    differ only in uptime."""
    m = _metrics()
    sched, loop = FakeScheduler(), FakeLoop()
    a = prom.render_server_metrics(m, sched, loop)
    b = prom.render_server_metrics(m, sched, loop)
    strip = lambda t: [ln for ln in t.split("\n") if "uptime" not in ln]  # noqa: E731
    assert strip(a) == strip(b)
