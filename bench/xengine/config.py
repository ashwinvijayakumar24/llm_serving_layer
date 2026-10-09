"""
Configs -> seeded request streams.

Two YAML files drive every run:

  configs/slo.yaml             the ONE frozen SLO (SPEC "Fixed parameters")
  configs/workloads/W<k>.yaml  one workload: loop type, the expected point
                               grid (`points`, single-key mappings such as
                               {concurrency: 16} or {rate: 4}), length/sharing
                               distributions, window sizes

THE REPRODUCIBILITY CONTRACT
----------------------------
SPEC: "same seed -> byte-identical request stream for every engine". The
stream for one (workload, point, rep) is a pure function of

    (workload YAML, point value, rep index, prompt renderer)

and never of the arm. `derive_seed` mixes the point and rep into the base seed
so that

  * every arm at (W, point, rep) gets the identical stream, and
  * different points and reps get DIFFERENT streams.

The second property is not cosmetic. With prefix caching on (vLLM's and
SGLang's default, and ours), re-sending rep 1's prompts as rep 2 would be served
largely from the cache left behind by rep 1, and rep 2 would measure a warm
cache rather than the workload. Distinct streams per rep make reps independent
without having to restart the server between them.

`request_stream_sha256` hashes exactly what goes on the wire (model, prompt,
max_tokens, ignore_eos, temperature), so "identical across arms" is checkable
from the artifacts alone.

PROMPT RENDERING
----------------
The workload generator works in token ids; the chat endpoint takes text. Two
renderers:

  hf_tokenizer  decode the ids with the checkpoint's own tokenizer. Prompt
                lengths then match the configured token counts closely (a
                decode/re-encode round trip is not exact, and the chat template
                adds its own header tokens). The default whenever a tokenizer is
                available.
  words         `bench.driver_common.prompt_text_from_ids` ("w123 w456 ...").
                Needs nothing, preserves prefix sharing at word boundaries, but
                one "word" is several real tokens, so prompts are ~2-3x longer
                than configured. A run rendered this way gets the validity
                reason `prompt_render_words_fallback`.

The renderer name is recorded in every artifact. Arms compared with each other
must share it; the stream hash enforces that.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.driver_common import prompt_text_from_ids, specs_from_prompts
from bench.loadgen import ArrivalProcess, LoadGenConfig, Phase, RequestSpec, arrival_schedule
from bench.workloads.generator import LengthSpec, WorkloadConfig, generate

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
WORKLOAD_DIR = CONFIG_DIR / "workloads"
SLO_PATH = CONFIG_DIR / "slo.yaml"

# Same model string on the wire for every engine. vLLM and SGLang are launched
# with `--served-model-name` set to this, and it is our server's default
# `model_id` (serving/server/app.py:MODEL_ID_DEFAULT), so request bodies are
# byte-identical across engines.
SERVED_MODEL_NAME = "llama-3.2-1b-instruct"

# Sent to EVERY engine, so request bodies stay byte-identical across arms. vLLM
# and SGLang answer with a final `choices: []` chunk carrying `usage`
# (prompt_tokens, and cached tokens with --enable-prompt-tokens-details /
# --enable-cache-report). Ours ignores the field (docs/xengine/ENGINE_FLAGS.md
# §3), so its server-side prompt token count is recorded as null.
STREAM_OPTIONS: dict[str, Any] = {"stream_options": {"include_usage": True}}


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict[str, Any]:
    """PyYAML (declared in the `bench` extra). Fails loudly if it is missing."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyYAML is required for bench/xengine configs: pip install -e '.[bench]'"
        ) from exc
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# SLO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SLO:
    ttft_ms: float
    tpot_ms: float
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {"ttft_ms": self.ttft_ms, "tpot_ms": self.tpot_ms, "source": self.source}


def load_slo(path: str | Path = SLO_PATH) -> SLO:
    d = load_yaml(path)
    for k in ("ttft_ms", "tpot_ms", "source"):
        if k not in d:
            raise ValueError(f"{path}: missing required key {k!r}")
    slo = SLO(float(d["ttft_ms"]), float(d["tpot_ms"]), str(d["source"]))
    if slo.ttft_ms <= 0 or slo.tpot_ms <= 0:
        raise ValueError(f"{path}: SLO thresholds must be positive, got {slo}")
    return slo


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


@dataclass
class WorkloadSpec:
    """One parsed workload YAML."""

    id: str
    loop: str  # "closed" | "open" | "derived"
    seed: int
    point_list: list[dict[str, Any]]
    requests: dict[str, Any]
    closed_loop: dict[str, Any] = field(default_factory=dict)
    open_loop: dict[str, Any] = field(default_factory=dict)
    request_timeout_s: float = 300.0
    kv_pool_tokens: int | None = None
    arms: list[str] = field(default_factory=list)
    blocked: bool = False
    blocked_reason: str | None = None
    title: str = ""
    config_path: str = ""
    config_sha256: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def sweep_param(self) -> str | None:
        return next(iter(self.point_list[0])) if self.point_list else None

    def points(self) -> list[dict[str, Any]]:
        return [dict(p) for p in self.point_list]


def _num(v: Any) -> int | float:
    f = float(v)
    return int(f) if f.is_integer() else f


def workload_path(workload_id: str) -> Path:
    return WORKLOAD_DIR / f"{workload_id}.yaml"


def load_workload(workload: str | Path) -> WorkloadSpec:
    """Accepts an id ("W1") or a path to a YAML file."""
    p = Path(workload)
    if not p.suffix:
        p = workload_path(str(workload))
    d = load_yaml(p)
    loop = str(d.get("loop", ""))
    if loop not in ("closed", "open", "derived"):
        raise ValueError(f"{p}: loop must be closed|open|derived, got {loop!r}")
    point_list = []
    for pt in d.get("points") or []:
        if not isinstance(pt, dict) or len(pt) != 1:
            raise ValueError(f"{p}: every point must be a single-key mapping, got {pt!r}")
        ((k, v),) = pt.items()
        point_list.append({str(k): _num(v)})
    spec = WorkloadSpec(
        id=str(d["id"]),
        loop=loop,
        seed=int(d.get("seed", 0)),
        point_list=point_list,
        requests=dict(d.get("requests") or {}),
        closed_loop=dict(d.get("closed_loop") or {}),
        open_loop=dict(d.get("open_loop") or {}),
        request_timeout_s=float(d.get("request_timeout_s", 300.0)),
        kv_pool_tokens=(int(d["kv_pool_tokens"]) if d.get("kv_pool_tokens") else None),
        arms=list(d.get("arms") or []),
        blocked=bool(d.get("blocked", False)),
        blocked_reason=d.get("blocked_reason"),
        title=str(d.get("title", "")),
        config_path=str(p),
        config_sha256=file_sha256(p),
        raw=d,
    )
    want = {"closed": "concurrency", "open": "rate"}.get(spec.loop)
    if want and any(next(iter(pt)) != want for pt in point_list):
        raise ValueError(f"{p}: every point of a {spec.loop}-loop workload is {{{want}: ...}}")
    if spec.loop != "derived" and not point_list:
        raise ValueError(f"{p}: `points` is empty")
    return spec


def generator_config(spec: WorkloadSpec, n_requests: int, seed: int) -> WorkloadConfig:
    """`requests:` block -> `bench.workloads.generator.WorkloadConfig`."""
    r = dict(spec.requests)
    prompt = LengthSpec(**r.pop("prompt", {}))
    output = LengthSpec(**r.pop("output", {}))
    return WorkloadConfig(
        n_requests=n_requests,
        seed=seed,
        prompt=prompt,
        output=output,
        name=f"xengine_{spec.id}",
        **r,
    )


def derive_seed(base_seed: int, workload_id: str, point: dict[str, Any], rep: int) -> int:
    """
    Per-(point, rep) seed, independent of the arm. See the module docstring for
    why reps must not reuse one stream when a prefix cache is on.
    """
    key = json.dumps(
        {"base": base_seed, "workload": workload_id, "point": point, "rep": rep},
        sort_keys=True,
    )
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


class WordsRenderer:
    name = "words"

    def render(self, token_ids: tuple[int, ...]) -> str:
        return prompt_text_from_ids(token_ids)

    def count(self, text: str) -> int | None:
        return None


class TokenizerRenderer:
    name = "hf_tokenizer"

    def __init__(self, tokenizer: Any, source: str = "") -> None:
        self.tokenizer = tokenizer
        self.source = source

    def render(self, token_ids: tuple[int, ...]) -> str:
        return str(self.tokenizer.decode(list(token_ids)))

    def count(self, text: str) -> int | None:
        enc = getattr(self.tokenizer, "encode", None)
        if enc is None:
            return None
        try:
            return len(enc(text, add_special_tokens=False))
        except TypeError:
            return len(enc(text))


def make_renderer(spec: str | None = "auto") -> WordsRenderer | TokenizerRenderer:
    """
    "words" -> WordsRenderer. A path -> that HF tokenizer. "auto" -> the
    tokenizer at $LLM_WEIGHTS_PATH if it loads, else words (and the run is
    flagged by the caller).
    """
    if spec == "words":
        return WordsRenderer()
    path = os.environ.get("LLM_WEIGHTS_PATH") if spec in (None, "auto") else spec
    if not path or not Path(path).exists():
        if spec not in (None, "auto"):
            raise FileNotFoundError(f"tokenizer path {path!r} does not exist")
        return WordsRenderer()
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path)
    except Exception as exc:  # noqa: BLE001 — fallback is recorded by the caller
        if spec not in (None, "auto"):
            raise RuntimeError(f"could not load tokenizer from {path}: {exc}") from exc
        return WordsRenderer()
    return TokenizerRenderer(tok, source=str(path))


# ---------------------------------------------------------------------------
# Request streams
# ---------------------------------------------------------------------------


@dataclass
class RequestStream:
    """What one (workload, point, rep) sends, decided before the run starts."""

    workload_id: str
    point: dict[str, Any]
    rep: int
    seed: int
    loop: str
    specs: list[RequestSpec]
    renderer: str
    sha256: str
    workload_fingerprint: str
    realized: dict[str, Any]
    lcfg: LoadGenConfig
    concurrency: int | None = None

    @property
    def n_requests(self) -> int:
        return len(self.specs)


def stream_sha256(specs: list[RequestSpec], cfg: LoadGenConfig) -> str:
    """Hash of exactly what goes on the wire, in send order."""
    h = hashlib.sha256()
    h.update(
        json.dumps(
            {
                "model": cfg.model,
                "ignore_eos": cfg.ignore_eos,
                "temperature": 0.0,
                "stream": True,
                "extra_body": cfg.extra_body,
            },
            sort_keys=True,
        ).encode()
    )
    for s in specs:
        h.update(b"\x00")
        h.update(json.dumps([s.prompt, s.max_tokens]).encode())
    return h.hexdigest()


def closed_loop_counts(spec: WorkloadSpec, concurrency: int) -> tuple[int, int, int]:
    c = spec.closed_loop
    warm = concurrency * int(c.get("warmup_per_worker", 2))
    meas = max(concurrency * int(c.get("measured_per_worker", 8)), int(c.get("min_measured", 32)))
    drain = concurrency * int(c.get("drain_per_worker", 2))
    return warm, meas, drain


def build_loadgen_config(
    spec: WorkloadSpec,
    point: dict[str, Any],
    seed: int,
    slo: SLO,
    url: str,
    model: str = SERVED_MODEL_NAME,
) -> LoadGenConfig:
    ol = spec.open_loop
    rate = float(point.get("rate", 1.0))
    return LoadGenConfig(
        url=url,
        model=model,
        rate_rps=rate,
        duration_s=float(ol.get("duration_s", 60.0)),
        warmup_s=float(ol.get("warmup_s", 15.0)),
        drain_s=float(ol.get("drain_s", 10.0)),
        process=str(ol.get("process", ArrivalProcess.POISSON)),
        seed=seed,
        slo_ttft_ms=slo.ttft_ms,
        slo_itl_ms=slo.tpot_ms,
        request_timeout_s=spec.request_timeout_s,
        max_dispatch_drift_ms=float(ol.get("max_dispatch_drift_ms", 50.0)),
        ignore_eos=True,
        name=f"xengine_{spec.id}",
        extra_body=json.loads(json.dumps(STREAM_OPTIONS)),
    )


def build_stream(
    spec: WorkloadSpec,
    point: dict[str, Any],
    rep: int,
    slo: SLO,
    url: str,
    renderer: WordsRenderer | TokenizerRenderer | None = None,
    model: str = SERVED_MODEL_NAME,
) -> RequestStream:
    """
    The full request list for one (workload, point, rep). Open loop: intended
    dispatch times from the same Poisson process bench/loadgen uses. Closed
    loop: no times — a worker sends when it is free — and phases by position.
    """
    if spec.loop == "derived" or spec.blocked:
        raise ValueError(f"workload {spec.id} is not directly runnable: {spec.blocked_reason}")
    renderer = renderer or WordsRenderer()
    seed = derive_seed(spec.seed, spec.id, point, rep)
    lcfg = build_loadgen_config(spec, point, seed, slo, url, model)

    concurrency = None
    if spec.loop == "open":
        times = arrival_schedule(lcfg.rate_rps, lcfg.horizon_s, lcfg.process, seed)
        n = len(times)
    else:
        concurrency = int(point["concurrency"])
        warm, meas, drain = closed_loop_counts(spec, concurrency)
        n = warm + meas + drain
        times = None

    wl = generate(generator_config(spec, max(n, 1), seed))
    prompts = [renderer.render(r.token_ids) for r in wl.requests[:n]]
    max_tokens = [r.max_tokens for r in wl.requests[:n]]

    if spec.loop == "open":
        specs = specs_from_prompts(lcfg, prompts, max_tokens, times)
    else:
        specs = []
        for i in range(n):
            phase = Phase.WARMUP if i < warm else (Phase.STEADY if i < warm + meas else Phase.DRAIN)
            specs.append(
                RequestSpec(
                    request_id=i,
                    intended_send_time=math.nan,
                    prompt=prompts[i],
                    max_tokens=max(1, max_tokens[i]),
                    phase=phase,
                )
            )

    realized = dict(wl.realized)
    counts = [c for c in (renderer.count(p) for p in prompts) if c is not None]
    if counts:
        realized["rendered_prompt_tokens_mean"] = sum(counts) / len(counts)
        realized["rendered_prompt_tokens_min"] = min(counts)
        realized["rendered_prompt_tokens_max"] = max(counts)
    return RequestStream(
        workload_id=spec.id,
        point=point,
        rep=rep,
        seed=seed,
        loop=spec.loop,
        specs=specs,
        renderer=renderer.name,
        sha256=stream_sha256(specs, lcfg),
        workload_fingerprint=wl.fingerprint(),
        realized=realized,
        lcfg=lcfg,
        concurrency=concurrency,
    )


def point_label(point: dict[str, Any]) -> str:
    """{"concurrency": 16} -> "concurrency16"; {"rate": 2.5} -> "rate2.5"."""
    return "_".join(f"{k}{_num(v)}" for k, v in sorted(point.items()))
