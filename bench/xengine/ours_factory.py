"""
`uvicorn --factory` entry point for OUR server that also reports which attention
backend actually loaded.

WHY THIS EXISTS
---------------
SPEC: "the active attention backend MUST be recorded and a PagedTorch fallback
marks the run invalid". `serving.server.app:build_default_app` tries
FlashInferBackend and silently falls back to PagedTorchBackend on ANY exception
(`except Exception: backend = None`, by design — R18). Nothing in /health,
/metrics or the server's stdout says which one won: the only startup line is
`config: kv_blocks=... preemption=... prefix_cache=... static_batching=...`.
So from outside the process, a FlashInfer run and a PagedTorch run look the
same until you compare their speed — which is the thing being measured.

This wrapper calls `build_default_app()` with NO arguments, exactly as
`uvicorn --factory serving.server.app:build_default_app` does, so the server is
the same program with the same configuration. It then reads
`app.state.scheduler.backend` (set by `create_app`) and prints one line that
`bench/xengine/engines.py:OursAdapter.detect_attention_backend` greps for:

    xengine: attention_backend=FlashInferBackend

serving/ is not modified (SPEC rule 4). If this wrapper is not used (the
adapter's `probe_backend=False`), the backend is recorded as "unknown" and the
run gets the validity reason `attention_backend_unverified`.
"""

from __future__ import annotations

from typing import Any

BACKEND_LOG_PREFIX = "xengine: attention_backend="
QUANT_LOG_PREFIX = "xengine: weight_quant="
QUANT_ENV = "XENGINE_OURS_QUANT"


def install_quant_loader(mode: str) -> dict[str, int]:
    """Make build_default_app load int8/int4 weights without modifying serving/.

    build_default_app does ``from engine.loader import load_weights_gpu`` at call
    time, so replacing the module attribute before the call swaps the loader. The
    replacement applies exactly the quantization of the engine's own
    ``load_weights_gpu_quant`` (same suffixes, same quantize functions); it is
    re-implemented here only because that function calls ``load_weights_gpu``
    through the module global, which would recurse into this patch.
    Returns a dict that is filled with the count of quantized linears on load.
    """
    if mode not in ("int8", "int4"):
        raise ValueError(f"{QUANT_ENV}={mode!r}: expected int8 or int4")
    import engine.loader as loader
    from engine.quant import QuantWeight, quantize_int4_group, quantize_int8_perchannel

    original = loader.load_weights_gpu
    stats = {"quantized_linears": 0}

    def load_quantized(weights_path: str, config: dict, device: str = "cuda:0") -> dict:
        weights = original(weights_path, config, device)
        for name in list(weights):
            if not name.endswith(loader._QUANTIZED_SUFFIXES):
                continue
            if mode == "int8":
                q, scale = quantize_int8_perchannel(weights[name])
                weights[name] = QuantWeight(q, scale, "int8")
            else:
                q, scale = quantize_int4_group(weights[name], group_size=128)
                weights[name] = QuantWeight(q, scale, "int4", 128)
            stats["quantized_linears"] += 1
        return weights

    loader.load_weights_gpu = load_quantized
    return stats


def backend_name_of(app: Any) -> str:
    sched = getattr(getattr(app, "state", None), "scheduler", None)
    backend = getattr(sched, "backend", None)
    return type(backend).__name__ if backend is not None else "unknown"


def build_app() -> Any:
    import os

    from serving.server.app import build_default_app

    quant = os.environ.get(QUANT_ENV)
    stats = install_quant_loader(quant) if quant else None
    app = build_default_app()
    print(f"{BACKEND_LOG_PREFIX}{backend_name_of(app)}", flush=True)
    if stats is not None:
        # Zero quantized linears means the patch did not take: the run would be
        # fp16 under an int8 label, so the harness marks it invalid.
        print(
            f"{QUANT_LOG_PREFIX}{quant} quantized_linears={stats['quantized_linears']}", flush=True
        )
    return app
