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


def backend_name_of(app: Any) -> str:
    sched = getattr(getattr(app, "state", None), "scheduler", None)
    backend = getattr(sched, "backend", None)
    return type(backend).__name__ if backend is not None else "unknown"


def build_app() -> Any:
    from serving.server.app import build_default_app

    app = build_default_app()
    print(f"{BACKEND_LOG_PREFIX}{backend_name_of(app)}", flush=True)
    return app
