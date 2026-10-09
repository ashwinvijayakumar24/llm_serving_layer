"""
Cross-engine harness (bench/xengine/): configs, request streams, adapters,
closed loop, artifact schema, and the CLI end to end against mock servers.

CPU-only. No vLLM, SGLang or GPU anywhere: the servers are tests/xengine_mock.py.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import time
from pathlib import Path

import httpx
import pytest

from bench.loadgen import LoadGenConfig, Phase, RequestSpec
from bench.xengine import SCHEMA
from bench.xengine import config as xc
from bench.xengine import engines as xe
from bench.xengine import run as xr
from bench.xengine.artifact import summarize, validate_artifact
from bench.xengine.closed_loop import inflight_check, run_closed_loop
from bench.xengine.hardware import (
    GpuSampler,
    capture_hardware,
    parse_cuda_banner,
    parse_nvidia_smi_query,
    publishable,
)
from tests.xengine_mock import serve_in_thread

SLO = xc.SLO(434.0, 27.8, "test")
URL = "http://x/v1/chat/completions"

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


def test_slo_yaml_matches_p2_results():
    slo = xc.load_slo()
    assert (slo.ttft_ms, slo.tpot_ms) == (434.0, 27.8)
    text = (Path(__file__).resolve().parents[1] / "results/p2/RESULTS.md").read_text()
    assert "**< 434 ms**" in text and "**< 27.8 ms**" in text
    assert "results/p2/RESULTS.md" in slo.source


@pytest.mark.parametrize("wid", ["W1", "W2", "W3", "W4"])
def test_workloads_load_with_single_key_points(wid):
    spec = xc.load_workload(wid)
    assert spec.points(), wid
    assert all(len(p) == 1 for p in spec.points())
    key = "concurrency" if spec.loop == "closed" else "rate"
    assert all(next(iter(p)) == key for p in spec.points())
    assert all(a in xe.ARMS for a in spec.arms)
    assert len(spec.config_sha256) == 64


def test_w1_grid_and_w4_pool():
    assert [p["concurrency"] for p in xc.load_workload("W1").points()] == [1, 2, 4, 8, 16, 32, 64]
    assert xc.load_workload("W4").kv_pool_tokens == 32768
    w3 = xc.load_workload("W3")
    assert w3.requests["sharing_rate"] == 0.8 and w3.requests["n_shared_prefixes"] == 1


def test_w5_points_to_int8_arm_and_int8_needs_pool():
    w5 = xc.load_workload("W5")
    assert w5.blocked and "bench-w5" in w5.blocked_reason
    arm = xe.get_arm("ours-int8")
    assert arm.blocked_reason is None and arm.matched
    assert dict(arm.env)["XENGINE_OURS_QUANT"] == "int8"
    adapter = xe.OursAdapter(probe_backend=True, python="py")
    with pytest.raises(ValueError):
        adapter.launch_spec(arm, "/w")
    spec = adapter.launch_spec(arm, "/w", kv_pool_tokens=32768)
    assert spec.env["SERVING_KV_BLOCKS"] == "2048"
    assert spec.flags["weight_quant"] == "int8"


def test_weight_quant_reason():
    ok = "xengine: weight_quant=int8 quantized_linears=112\n"
    assert xr.weight_quant_reason("int8", ok) is None
    assert "unverified" in xr.weight_quant_reason("int8", "")
    assert "unverified" in xr.weight_quant_reason(
        "int8", "xengine: weight_quant=int8 quantized_linears=0"
    )


def test_bad_point_shape_rejected(tmp_path):
    p = tmp_path / "W9.yaml"
    p.write_text("id: W9\nloop: open\npoints:\n  - {rate: 1, concurrency: 2}\nrequests: {}\n")
    with pytest.raises(ValueError, match="single-key"):
        xc.load_workload(p)


# ---------------------------------------------------------------------------
# Request streams: same seed -> identical stream for every arm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wid,point", [("W1", {"concurrency": 4}), ("W3", {"rate": 2})])
def test_same_seed_identical_stream_across_arms(wid, point):
    spec = xc.load_workload(wid)
    streams = []
    for arm_id in ("ours", "vllm-eager", "sglang-noradix"):
        adapter = xe.adapter_for(xe.get_arm(arm_id).engine, python="py")
        base = f"http://127.0.0.1:{8000 + len(streams)}"  # different servers
        streams.append(
            xc.build_stream(spec, point, 1, SLO, adapter.chat_url(base), None, adapter.model_name)
        )
    a = streams[0]
    for b in streams[1:]:
        assert b.sha256 == a.sha256
        assert [s.prompt for s in b.specs] == [s.prompt for s in a.specs]
        assert [s.max_tokens for s in b.specs] == [s.max_tokens for s in a.specs]
        assert [s.intended_send_time for s in b.specs] == [
            s.intended_send_time for s in a.specs
        ] or wid == "W1"  # closed: NaN times
        assert b.seed == a.seed


def test_reps_and_points_get_distinct_streams():
    spec = xc.load_workload("W3")
    r1 = xc.build_stream(spec, {"rate": 2}, 1, SLO, URL)
    r2 = xc.build_stream(spec, {"rate": 2}, 2, SLO, URL)
    p4 = xc.build_stream(spec, {"rate": 4}, 1, SLO, URL)
    assert len({r1.sha256, r2.sha256, p4.sha256}) == 3
    assert r1.specs[0].prompt != r2.specs[0].prompt  # no cross-rep prefix-cache reuse


def test_stream_carries_stream_options_into_hash():
    spec = xc.load_workload("W2")
    st = xc.build_stream(spec, {"rate": 1}, 1, SLO, URL)
    assert st.lcfg.extra_body == {"stream_options": {"include_usage": True}}
    other = copy.deepcopy(st.lcfg)
    other.extra_body = {}
    assert xc.stream_sha256(st.specs, other) != st.sha256


def test_closed_loop_stream_phases():
    spec = xc.load_workload("W1")
    st = xc.build_stream(spec, {"concurrency": 4}, 1, SLO, URL)
    warm, meas, drain = xc.closed_loop_counts(spec, 4)
    phases = [s.phase for s in st.specs]
    assert phases.count(Phase.WARMUP) == warm == 8
    assert phases.count(Phase.STEADY) == meas == 32
    assert phases.count(Phase.DRAIN) == drain == 8
    assert all(s.max_tokens == 128 for s in st.specs)


def test_tokenizer_renderer_and_counts():
    class FakeTok:
        def decode(self, ids):
            return " ".join(f"t{i}" for i in ids)

        def encode(self, text, add_special_tokens=True):
            return text.split()

    spec = xc.load_workload("W1")
    st = xc.build_stream(spec, {"concurrency": 1}, 1, SLO, URL, xc.TokenizerRenderer(FakeTok()))
    assert st.renderer == "hf_tokenizer"
    assert st.realized["rendered_prompt_tokens_mean"] == 512
    assert xc.make_renderer("words").name == "words"


# ---------------------------------------------------------------------------
# Launch commands per arm
# ---------------------------------------------------------------------------


def _cmd(arm_id: str, kv: int | None = None, monkeypatch=None) -> list[str]:
    arm = xe.get_arm(arm_id)
    return xe.adapter_for(arm.engine, python="PY").launch_spec(arm, "/w", "127.0.0.1", 9000, kv).cmd


def _flag_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_vllm_launch_flags():
    base = _cmd("vllm")
    assert base[:4] == ["PY", "-m", "vllm.entrypoints.cli.main", "serve"]
    assert _flag_value(base, "--model") == "/w"
    assert _flag_value(base, "--dtype") == "float16"
    assert _flag_value(base, "--seed") == "0"
    assert _flag_value(base, "--served-model-name") == xc.SERVED_MODEL_NAME
    assert "--enable-prompt-tokens-details" in base
    for diag in (
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--compilation-config",
        "--num-gpu-blocks-override",
    ):
        assert diag not in base
    assert "--enforce-eager" in _cmd("vllm-eager")
    ng = _cmd("vllm-nograph")
    assert json.loads(_flag_value(ng, "--compilation-config")) == {"cudagraph_mode": "NONE"}
    assert "--enforce-eager" not in ng
    assert "--no-async-scheduling" in _cmd("vllm-noasync")
    assert "--no-enable-prefix-caching" in _cmd("vllm-noprefix")
    m = _cmd("vllm-matched", kv=32768)
    assert _flag_value(m, "--max-num-seqs") == "32"
    assert _flag_value(m, "--num-gpu-blocks-override") == "2048"
    assert _flag_value(m, "--block-size") == "16"
    with pytest.raises(ValueError, match="vllm-matched"):
        _cmd("vllm-matched")


def test_sglang_launch_flags():
    base = _cmd("sglang")
    assert base[:3] == ["PY", "-m", "sglang.launch_server"]
    assert _flag_value(base, "--model-path") == "/w"
    assert _flag_value(base, "--dtype") == "float16"
    assert _flag_value(base, "--random-seed") == "0"
    assert "--enable-metrics" in base and "--enable-cache-report" in base
    assert "--disable-radix-cache" in _cmd("sglang-noradix")
    assert "--disable-overlap-schedule" in _cmd("sglang-nooverlap")
    eager = _cmd("sglang-eager")
    assert "--disable-decode-cuda-graph" in eager and "--disable-prefill-cuda-graph" in eager
    assert "--disable-cuda-graph" not in eager  # deprecated spelling
    assert _flag_value(_cmd("sglang-lpm"), "--schedule-policy") == "lpm"
    assert _flag_value(_cmd("sglang", kv=32768), "--max-total-tokens") == "32768"
    assert _flag_value(_cmd("sglang", kv=32770), "--max-total-tokens") == "32768"  # 16-aligned


def test_ours_launch_spec_and_env():
    arm = xe.get_arm("ours")
    ls = xe.OursAdapter(python="PY").launch_spec(arm, "/w", "127.0.0.1", 9000, 32768)
    assert ls.cmd[:4] == ["PY", "-m", "uvicorn", "bench.xengine.ours_factory:build_app"]
    assert "--factory" in ls.cmd
    assert ls.env["LLM_WEIGHTS_PATH"] == "/w"
    assert ls.env["SERVING_KV_BLOCKS"] == "2048"
    assert ls.flags["prefix_cache"] is True and ls.flags["kv_pool_tokens"] == 32768
    plain = xe.OursAdapter(probe_backend=False, python="PY").launch_spec(arm, "/w")
    assert "serving.server.app:build_default_app" in plain.cmd
    nop = xe.OursAdapter(python="PY").launch_spec(xe.get_arm("ours-noprefix"), "/w")
    assert nop.env["SERVING_PREFIX_CACHE"] == "0" and nop.flags["prefix_cache"] is False


def test_server_python_from_env(monkeypatch):
    monkeypatch.setenv("XENGINE_SERVER_PY", "/envs/vllm/bin/python")
    assert _cmd_default("vllm")[0] == "/envs/vllm/bin/python"
    monkeypatch.delenv("XENGINE_SERVER_PY")
    assert _cmd_default("sglang")[0] == sys.executable


def _cmd_default(arm_id: str) -> list[str]:
    arm = xe.get_arm(arm_id)
    return xe.adapter_for(arm.engine).launch_spec(arm, "/w").cmd


def test_spec_arm_table_is_complete():
    assert set(xe.ARMS) == {
        "ours",
        "ours-noprefix",
        "ours-int8",
        "vllm",
        "vllm-eager",
        "vllm-nograph",
        "vllm-noasync",
        "vllm-noprefix",
        "vllm-matched",
        "sglang",
        "sglang-noradix",
        "sglang-nooverlap",
        "sglang-eager",
        "sglang-lpm",
    }
    assert {a.id for a in xe.ARMS.values() if not a.diagnostic} == {"ours", "vllm", "sglang"}


# ---------------------------------------------------------------------------
# Counters: None (not 0) when absent
# ---------------------------------------------------------------------------


def test_prometheus_parse_labels_and_skips():
    text = (
        "# HELP x\n"
        'vllm:num_preemptions_total{model_name="m",engine="0"} 3\n'
        'sglang:prefill_effective_tokens_total{mode="device_hit",model_name="m"} 20\n'
        'sglang:prefill_effective_tokens_total{mode="miss",model_name="m"} 80\n'
        'h_bucket{le="1"} 5\nh_count 5\nfoo_created 1.7e9\n'
    )
    d = xe.parse_prometheus(text)
    assert d["vllm:num_preemptions_total"] == 3
    assert d["sglang:prefill_effective_tokens_total"] == 100
    assert d['sglang:prefill_effective_tokens_total{mode="device_hit"}'] == 20
    assert "h_bucket" not in d and "foo_created" not in d and d["h_count"] == 5


def test_counters_none_when_absent():
    ours = xe.OursAdapter(python="PY")
    before = {"scheduler": {"preemptions_total": 1}}
    after = {"scheduler": {"preemptions_total": 1}}  # prefix cache off: no cache_*
    sc = ours.server_counters(before, after)
    assert sc["preemptions"] == 0.0  # present and unchanged -> 0
    assert sc["evictions"] is None and sc["prefix_hit_rate"] is None
    assert xe.OursAdapter(python="PY").server_counters({}, {})["preemptions"] is None

    for engine in ("vllm", "sglang"):
        sc = xe.adapter_for(engine, python="PY").server_counters({}, {"_error": "HTTP 404"})
        assert sc["preemptions"] is None and sc["evictions"] is None
        assert sc["prefix_hit_rate"] is None
        assert isinstance(sc["raw"], dict)
    # vLLM never exposes evictions, even when everything else is there
    v = xe.adapter_for("vllm", python="PY").server_counters(
        {"vllm:num_preemptions_total": 0.0}, {"vllm:num_preemptions_total": 2.0}
    )
    assert v["preemptions"] == 2.0 and v["evictions"] is None


def test_sglang_gauge_is_not_used_for_hit_rate():
    sc = xe.adapter_for("sglang", python="PY").server_counters({}, {"sglang:cache_hit_rate": 0.99})
    assert sc["prefix_hit_rate"] is None


def test_counters_from_mock_servers():
    for flavor, want_rate, want_ev in (
        ("vllm", 0.75, None),
        ("sglang", 0.30, 7.0 * 4),
        ("ours", 0.75, 2.0),
    ):
        srv = serve_in_thread(flavor=flavor)
        try:
            ad = xe.adapter_for(flavor, python="PY")
            before = ad.scrape_counters(srv.base_url)
            for _ in range(4):
                srv.app.state.mock.requests += 1
            after = ad.scrape_counters(srv.base_url)
            sc = ad.server_counters(before, after)
            assert sc["prefix_hit_rate"] == pytest.approx(want_rate), flavor
            assert sc["preemptions"] == 1.0, flavor
            assert sc["evictions"] == want_ev, flavor
        finally:
            srv.stop()


def test_counters_none_from_bare_and_prefix_off_servers():
    for kw, engine in (
        (dict(flavor="vllm", bare=True), "vllm"),
        (dict(flavor="sglang", bare=True), "sglang"),
        (dict(flavor="ours", prefix_off=True), "ours"),
    ):
        srv = serve_in_thread(**kw)
        try:
            ad = xe.adapter_for(engine, python="PY")
            sc = ad.server_counters(
                ad.scrape_counters(srv.base_url), ad.scrape_counters(srv.base_url)
            )
            assert sc["prefix_hit_rate"] is None and sc["evictions"] is None
            if engine != "ours":
                assert sc["preemptions"] is None
        finally:
            srv.stop()


def test_version_queries():
    srv = serve_in_thread(flavor="vllm", version="9.9.9")
    try:
        assert xe.adapter_for("vllm", python="PY").query_version(srv.base_url) == "9.9.9"
    finally:
        srv.stop()
    srv = serve_in_thread(flavor="sglang", version="0.5.21")  # /version 404 -> server_info
    try:
        assert xe.adapter_for("sglang", python="PY").query_version(srv.base_url) == "0.5.21"
    finally:
        srv.stop()


# ---------------------------------------------------------------------------
# Attention backend / log greps
# ---------------------------------------------------------------------------


def test_backend_and_config_detection_from_logs():
    ours = xe.OursAdapter(python="PY")
    log = (
        "config: kv_blocks=2048 preemption=PreemptionPolicy.RECOMPUTE prefix_cache=OFF "
        "static_batching=False\nxengine: attention_backend=FlashInferBackend\n"
    )
    assert ours.detect_attention_backend(log) == "flashinfer"
    assert (
        ours.detect_attention_backend("xengine: attention_backend=PagedTorchBackend")
        == "paged_torch"
    )
    assert ours.detect_attention_backend("config: kv_blocks=1 ...") is None
    cfg = ours.config_from_log(log)
    assert cfg == {
        "kv_blocks": 2048,
        "preemption": "PreemptionPolicy.RECOMPUTE",
        "prefix_cache": False,
        "static_batching": False,
    }
    assert xr.ours_config_reasons({"prefix_cache": True}, cfg)[0].startswith("config_mismatch")
    assert xr.ours_config_reasons({"prefix_cache": False, "kv_blocks": 2048}, cfg) == []

    v = xe.adapter_for("vllm", python="PY")
    vlog = (
        "INFO Using FlashAttention backend.\n"
        "INFO Chunked prefill is enabled with max_num_batched_tokens=8192.\n"
        "INFO non-default args: {'max_num_seqs': 32}\nINFO GPU KV cache size: 1,234,560 tokens\n"
    )
    assert v.detect_attention_backend(vlog) == "flashattention"
    assert v.resolved_defaults(vlog) == {
        "max_num_seqs": 32,
        "max_num_batched_tokens": 8192,
        "kv_cache_tokens": 1234560,
    }
    s = xe.adapter_for("sglang", python="PY")
    slog = (
        "server_args=ServerArgs(attention_backend='flashinfer', max_running_requests=None, "
        "chunked_prefill_size=8192, schedule_policy='fcfs')\nmax_total_num_tokens=500000\n"
    )
    assert s.detect_attention_backend(slog) == "flashinfer"
    assert s.resolved_defaults(slog)["max_total_num_tokens"] == 500000
    assert s.resolved_defaults(slog)["schedule_policy"] == "fcfs"


def test_preflight_parse_and_resolution():
    ok = xe.parse_preflight_output('noise\nXENGINE_PREFLIGHT {"backend": "flashinfer"}\n', 0)
    assert ok["backend"] == "flashinfer"
    bad = xe.parse_preflight_output("", 1, "Traceback\nImportError: no flashinfer")
    assert bad["backend"].startswith("paged_torch(")
    arm = xe.get_arm("ours")
    assert xr.resolve_attention_backend(arm, ok, "flashinfer") == ("flashinfer", [])
    assert xr.resolve_attention_backend(arm, ok, None) == ("flashinfer", [])
    b, r = xr.resolve_attention_backend(arm, None, None)
    assert b == "unknown" and r[0].startswith("attention_backend_unverified")
    b, r = xr.resolve_attention_backend(arm, bad, None)
    assert b.startswith("paged_torch(") and r[0].startswith("attention_backend_not_flashinfer")
    b, r = xr.resolve_attention_backend(arm, ok, "paged_torch")  # OOM-only fallback
    assert b == "paged_torch" and r
    assert xr.resolve_attention_backend(xe.get_arm("vllm"), None, "flash_attn") == (
        "flash_attn",
        [],
    )


def test_preflight_runs_in_server_interpreter(tmp_path):
    """No FlashInfer and no GPU here: the pre-flight must report paged_torch(...)."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "num_attention_heads": 32,
                "hidden_size": 2048,
                "num_hidden_layers": 16,
                "num_key_value_heads": 8,
                "head_dim": 64,
            }
        )
    )
    res = xe.OursAdapter(python=sys.executable).preflight_attention_backend(
        str(tmp_path), timeout_s=120
    )
    assert res["backend"].startswith("paged_torch(") or res["backend"] == "flashinfer"


# ---------------------------------------------------------------------------
# Closed loop keeps exactly N in flight
# ---------------------------------------------------------------------------


class CountingSSE(httpx.AsyncBaseTransport):
    def __init__(self, token_delay: float = 0.003) -> None:
        self.inflight = 0
        self.peak = 0
        self.delay = token_delay

    async def handle_async_request(self, request):
        await request.aread()
        body = json.loads(request.content)
        me = self

        async def gen():
            # Counted from first byte to the [DONE] line. Not via try/finally:
            # httpx does not close a mock async-generator body promptly, so a
            # finally-decrement would run late and overcount.
            me.inflight += 1
            me.peak = max(me.peak, me.inflight)
            for i in range(body["max_tokens"]):
                await asyncio.sleep(me.delay)
                obj = {"choices": [{"delta": {"content": f"t{i}"}, "finish_reason": None}]}
                yield f"data: {json.dumps(obj)}\n\n".encode()
            yield b'data: {"choices": [{"delta": {}, "finish_reason": "length"}]}\n\n'
            me.inflight -= 1
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=gen())


def _closed_specs(warm: int, meas: int, drain: int) -> list[RequestSpec]:
    out = []
    for i in range(warm + meas + drain):
        ph = Phase.WARMUP if i < warm else Phase.STEADY if i < warm + meas else Phase.DRAIN
        out.append(RequestSpec(i, float("nan"), f"p{i}", 5 + (i % 3), ph))
    return out


@pytest.mark.parametrize("n", [1, 3, 8])
def test_closed_loop_keeps_exactly_n_in_flight(n):
    transport = CountingSSE()
    cfg = LoadGenConfig(url=URL, inflight_sample_interval_s=0.002)
    specs = _closed_specs(2 * n, 6 * n, 2 * n)

    async def go():
        async with httpx.AsyncClient(transport=transport) as c:
            return await run_closed_loop(cfg, specs, n, client=c)

    run = asyncio.run(go())
    assert transport.peak == n  # never more than N
    assert len(run.results) == len(specs)
    assert [r.spec.request_id for r in run.results] == list(range(len(specs)))
    chk = inflight_check(run, n)
    assert chk["held"], chk
    assert chk["min_inflight"] == n == chk["max_inflight"]  # never fewer either
    assert all(r.outcome == "completed" for r in run.results)
    # latency origin = worker-free time, so drift is harness overhead only
    assert max(r.dispatch_drift_ms for r in run.results) < 50


def test_closed_loop_flags_when_n_not_held():
    transport = CountingSSE()
    cfg = LoadGenConfig(url=URL, inflight_sample_interval_s=0.002)
    specs = _closed_specs(0, 6, 0)  # no drain: tail runs below N

    async def go():
        async with httpx.AsyncClient(transport=transport) as c:
            return await run_closed_loop(cfg, specs, 4, client=c)

    chk = inflight_check(asyncio.run(go()), 4)
    assert not chk["held"] and "in-flight" in chk["reason"]


# ---------------------------------------------------------------------------
# Artifact schema
# ---------------------------------------------------------------------------


def _minimal_artifact(arm="vllm") -> dict:
    eng = xe.get_arm(arm).engine
    return {
        "schema": SCHEMA,
        "arm": arm,
        "engine": {"name": eng, "version": "1", "launch_cmd": ["x"], "flags": {}},
        "workload": {
            "id": "W1",
            "config_path": "c",
            "config_sha256": "a" * 64,
            "seed": 1,
            "point": {"concurrency": 4},
        },
        "rep": 1,
        "hardware": {
            "gpu_name": None,
            "gpu_count": None,
            "driver": None,
            "cuda": None,
            "node": "n",
            "slurm_job_id": None,
        },
        "provenance": {
            "repo_sha": None,
            "repo_dirty": None,
            "started_at": "t",
            "harness_version": "0",
        },
        "slo": {"ttft_ms": 434.0, "tpot_ms": 27.8, "source": "s"},
        "validity": {"valid": True, "reasons": [], "attention_backend": None},
        "metrics": {
            "ttft_ms": summarize([1.0, 2.0]),
            "itl_ms": summarize([]),
            "tpot_ms": summarize([3.0]),
            "e2e_ms": summarize([4.0]),
            "output_tok_s": 1.0,
            "request_rps": 1.0,
            "goodput_rps": 1.0,
            "slo_attainment": 1.0,
            "completed": 1,
            "failed": 0,
        },
        "server_counters": {
            "preemptions": None,
            "evictions": None,
            "prefix_hit_rate": 0.5,
            "raw": {},
        },
        "anomalies": [{"kind": "k", "detail": "d"}],
        "samples": {"ttft_ms": [1.0, 2.0], "itl_ms": [], "tpot_ms": [3.0]},
    }


def test_validator_accepts_minimal_and_rejects_mutations():
    assert validate_artifact(_minimal_artifact()) == []
    mutations = {
        "extra top-level": lambda d: d.update(extra=1),
        "missing samples": lambda d: d.pop("samples"),
        "bad schema": lambda d: d.update(schema="x"),
        "unknown arm": lambda d: d.update(arm="trtllm"),
        "engine mismatch": lambda d: d["engine"].update(name="sglang"),
        "two-key point": lambda d: d["workload"].update(point={"rate": 1, "concurrency": 2}),
        "counter zero-string": lambda d: d["server_counters"].update(evictions="0"),
        "p50 with n=0": lambda d: d["metrics"]["itl_ms"].update(p50=0.0),
        "null p50 with n>0": lambda d: d["metrics"]["ttft_ms"].update(p50=None),
        "sample count": lambda d: d["samples"]["ttft_ms"].append(5.0),
        "valid with reasons": lambda d: d["validity"].update(reasons=["x"]),
        "invalid no reason": lambda d: d["validity"].update(valid=False),
        "rep 0": lambda d: d.update(rep=0),
        "attainment > 1": lambda d: d["metrics"].update(slo_attainment=1.5),
        "extra sample key": lambda d: d["samples"].update(e2e_ms=[]),
    }
    for name, mut in mutations.items():
        d = _minimal_artifact()
        mut(d)
        assert validate_artifact(d), name


def test_validator_ours_requires_flashinfer_when_valid():
    d = _minimal_artifact("ours")
    d["validity"]["attention_backend"] = "unknown"
    assert any("flashinfer" in e for e in validate_artifact(d))
    d["validity"].update(valid=False, reasons=["attention_backend_unverified: x"])
    assert validate_artifact(d) == []


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


def test_hardware_parsers_and_graceful_absence(monkeypatch):
    rows = parse_nvidia_smi_query("NVIDIA H200, 570.86.15\nNVIDIA H200, 570.86.15\n")
    assert rows == [{"name": "NVIDIA H200", "driver": "570.86.15"}] * 2
    assert parse_cuda_banner("| Driver Version: 570.86.15   CUDA Version: 12.8 |") == "12.8"
    assert parse_cuda_banner(None) is None
    assert GpuSampler.parse("1980, 61, 350.5") == {
        "sm_clock_mhz": 1980.0,
        "temp_c": 61.0,
        "power_w": 350.5,
    }
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("bench.xengine.hardware.shutil.which", lambda _: None)
    hw = capture_hardware()
    assert set(hw) >= {"gpu_name", "gpu_count", "driver", "cuda", "node", "slurm_job_id"}
    assert hw["driver"] is None and hw["slurm_job_id"] is None and hw["node"]
    ok, why = publishable(hw)
    assert not ok and any("SLURM" in w for w in why)
    s = GpuSampler(0.01)
    assert not s.available and s.start().stop() == []


# ---------------------------------------------------------------------------
# Server process lifecycle
# ---------------------------------------------------------------------------


def test_server_process_lifecycle(tmp_path):
    port = xr.free_port()
    spec = xe.LaunchSpec(
        cmd=[
            sys.executable,
            "-m",
            "tests.xengine_mock",
            "--port",
            str(port),
            "--flavor",
            "ours",
            "--print",
            "xengine: attention_backend=FlashInferBackend",
        ],
        env={},
        flags={},
    )
    log = tmp_path / "server.log"
    srv = xe.ServerProcess(spec, log, f"http://127.0.0.1:{port}/health").start()
    try:
        assert srv.wait_healthy(timeout_s=60, poll_s=0.2) > 0
        assert srv.alive()
    finally:
        srv.stop(grace_s=10)
    assert not srv.alive()
    text = log.read_text()
    assert "# xengine launch:" in text
    assert xe.OursAdapter(python="PY").detect_attention_backend(text) == "flashinfer"


def test_server_process_death_is_reported(tmp_path):
    spec = xe.LaunchSpec(
        cmd=[sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], env={}, flags={}
    )
    srv = xe.ServerProcess(spec, tmp_path / "s.log", "http://127.0.0.1:9/health").start()
    with pytest.raises(xe.ServerStartError, match="exited with code 3"):
        srv.wait_healthy(timeout_s=30, poll_s=0.1)
    srv.stop()
    assert "boom" in (tmp_path / "s.log").read_text()


# ---------------------------------------------------------------------------
# CLI end to end against the mock server (--url mode)
# ---------------------------------------------------------------------------

MINI_CLOSED = """
id: W8
loop: closed
seed: 11
points:
  - {concurrency: 1}
  - {concurrency: 3}
requests:
  structure: zero
  prompt: {dist: fixed, mean: 8}
  output: {dist: fixed, mean: 6}
closed_loop: {warmup_per_worker: 1, measured_per_worker: 4, min_measured: 4,
              drain_per_worker: 2}
request_timeout_s: 30
"""

MINI_OPEN = """
id: W9
loop: open
seed: 12
points:
  - {rate: 40}
requests:
  structure: system
  sharing_rate: 0.8
  n_shared_prefixes: 1
  shared_prefix_tokens: 16
  prompt: {dist: fixed, mean: 24}
  output: {dist: fixed, mean: 4}
open_loop: {warmup_s: 0.2, duration_s: 0.6, drain_s: 0.2, process: poisson}
request_timeout_s: 30
"""


def _run_cli(tmp_path, yaml_text, name, arm, flavor, reps=2, extra=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    wl = tmp_path / f"{name}.yaml"
    wl.write_text(yaml_text)
    out = tmp_path / "results"
    srv = serve_in_thread(flavor=flavor)
    try:
        rc = xr.main(
            [
                "--arm",
                arm,
                "--workload",
                str(wl),
                "--reps",
                str(reps),
                "--out",
                str(out),
                "--url",
                srv.base_url,
                "--renderer",
                "words",
                "--gpu-sample-interval",
                "0",
                *extra,
            ]
        )
        bodies = list(srv.app.state.mock.bodies)
    finally:
        srv.stop()
    return rc, out, bodies


def test_cli_closed_loop_end_to_end(tmp_path):
    from bench.xengine import render

    rc, out, bodies = _run_cli(tmp_path, MINI_CLOSED, "W8", "vllm", "vllm")
    paths = sorted((out / "W8" / "vllm").glob("*.json"))
    assert [p.name for p in paths] == [
        "concurrency1_rep1.json",
        "concurrency1_rep2.json",
        "concurrency3_rep1.json",
        "concurrency3_rep2.json",
    ]
    assert (out / "W8" / "vllm" / "engine_version.txt").exists()
    for p in paths:
        d = json.loads(p.read_text())
        assert validate_artifact(d) == [], p
        assert render.validate_artifact(d) == [], p
        assert d["workload"]["loop"] == "closed"
        assert "CLOSED LOOP" in d["workload"]["notes"][0]
        assert d["engine"]["version"] == "0.0.0-mock"
        assert d["engine"]["launch_cmd"] == []
        assert d["metrics"]["completed"] == d["metrics"]["steady_requests"] > 0
        assert d["server_counters"]["evictions"] is None  # vLLM: never 0
        assert d["server_counters"]["prefix_hit_rate"] == pytest.approx(0.75)
        assert d["workload"]["server_usage"]["prompt_tokens"]["min"] > 0
        assert d["validity"]["checks"]["inflight"]["held"]
        # words renderer -> run invalid, with a coded reason
        assert not d["validity"]["valid"]
        assert any(r.startswith("prompt_render_words_fallback") for r in d["validity"]["reasons"])
    assert rc == 2
    assert all(
        b["stream_options"] == {"include_usage": True}
        and b["ignore_eos"] is True
        and b["temperature"] == 0.0
        and b["stream"] is True
        for b in bodies
    )
    assert {b["model"] for b in bodies} == {xc.SERVED_MODEL_NAME}

    # appending: a second invocation writes rep3/rep4 with new seeds
    rc2, _, _ = _run_cli(
        tmp_path, MINI_CLOSED, "W8", "vllm", "vllm", reps=1, extra=["--points", "1"]
    )
    reps = sorted(p.name for p in (out / "W8" / "vllm").glob("concurrency1_rep*.json"))
    assert reps[-1] == "concurrency1_rep3.json"
    seeds = {json.loads((out / "W8" / "vllm" / r).read_text())["workload"]["seed"] for r in reps}
    assert len(seeds) == 3


def test_cli_open_loop_identical_stream_across_engines(tmp_path):
    rc_v, out_v, bodies_v = _run_cli(tmp_path / "v", MINI_OPEN, "W9", "vllm", "vllm", reps=1)
    rc_s, out_s, bodies_s = _run_cli(tmp_path / "s", MINI_OPEN, "W9", "sglang", "sglang", reps=1)
    rc_o, out_o, bodies_o = _run_cli(tmp_path / "o", MINI_OPEN, "W9", "ours", "ours", reps=1)
    dv = json.loads((out_v / "W9/vllm/rate40_rep1.json").read_text())
    ds = json.loads((out_s / "W9/sglang/rate40_rep1.json").read_text())
    do = json.loads((out_o / "W9/ours/rate40_rep1.json").read_text())
    for d in (dv, ds, do):
        assert validate_artifact(d) == []
        assert d["workload"]["loop"] == "open"
    assert (
        dv["workload"]["request_stream_sha256"]
        == ds["workload"]["request_stream_sha256"]
        == do["workload"]["request_stream_sha256"]
    )

    def key(b):
        return json.dumps(b, sort_keys=True)

    assert sorted(map(key, bodies_v)) == sorted(map(key, bodies_s)) == sorted(map(key, bodies_o))
    # ours: no pre-flight verdict (no model path) and no log -> unknown, invalid
    assert do["validity"]["attention_backend"] == "unknown"
    assert any(r.startswith("attention_backend_unverified") for r in do["validity"]["reasons"])
    assert do["workload"]["server_usage"]["prompt_tokens"] is None
    assert ds["server_counters"]["evictions"] is not None  # SGLang exposes it
    assert do["server_counters"]["evictions"] is not None  # ours JSON snapshot


def test_cli_dry_run_prints_launch(tmp_path, capsys):
    rc = xr.main(
        [
            "--arm",
            "sglang-eager",
            "--workload",
            "W1",
            "--model-path",
            str(tmp_path),
            "--out",
            str(tmp_path / "r"),
            "--dry-run",
            "--port",
            "9123",
        ]
    )
    assert rc == 0
    spec = json.loads(capsys.readouterr().out)
    assert "--disable-decode-cuda-graph" in spec["launch_cmd"]
    assert "9123" in spec["launch_cmd"]


def test_rep_spread_written_back(tmp_path):
    arts = []
    for i, tok in enumerate((1000.0, 1500.0, 700.0), start=1):
        d = _minimal_artifact()
        d["rep"] = i
        d["metrics"]["output_tok_s"] = tok
        p = tmp_path / f"concurrency4_rep{i}.json"
        p.write_text(json.dumps(d))
        arts.append(p)
    spread = xr.apply_rep_spread(arts, 0.10)
    assert [a["kind"] for a in spread] == ["rep_spread_output_tok_s"]
    for p in arts:
        d = json.loads(p.read_text())
        assert validate_artifact(d) == []
        assert {"k", "rep_spread_output_tok_s"} == {a["kind"] for a in d["anomalies"]}
    xr.apply_rep_spread(arts, 0.10)  # idempotent
    assert len(json.loads(arts[0].read_text())["anomalies"]) == 2


def test_wall_clock_budget():
    """Guard: the mini end-to-end workloads stay fast enough for CI."""
    t = time.perf_counter()
    spec = xc.load_workload("W1")
    xc.build_stream(spec, {"concurrency": 64}, 1, SLO, URL)
    assert time.perf_counter() - t < 10
