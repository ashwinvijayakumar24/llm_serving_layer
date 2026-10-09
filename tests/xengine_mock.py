"""
A fake OpenAI-compatible streaming server for the xengine harness tests.

One ASGI app, three `/metrics` flavours so each adapter's scraper is exercised
against the shape its real engine produces:

  ours    JSON, the subset of serving/server/app.py's /metrics the adapter reads
  vllm    Prometheus text with vllm:* names (from engines.py's VERIFY block)
  sglang  Prometheus text with sglang:* names

Counters advance with traffic so before/after deltas are non-trivial. The
`prefix_off` / `bare` switches drop metrics, to check that an absent metric
becomes None and not 0.

Usable in-process (`serve_in_thread`) or as a subprocess
(`python3 -m tests.xengine_mock --port P --flavor vllm`) for the lifecycle test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import threading
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse


class MockState:
    def __init__(self) -> None:
        self.requests = 0
        self.tokens = 0
        self.inflight = 0
        self.peak_inflight = 0
        self.bodies: list[dict[str, Any]] = []


def make_app(
    flavor: str = "vllm",
    ttft_s: float = 0.005,
    token_delay_s: float = 0.002,
    prefix_off: bool = False,
    bare: bool = False,
    version: str = "0.0.0-mock",
) -> FastAPI:
    app = FastAPI()
    st = MockState()
    app.state.mock = st

    def chunk(content: str | None, finish: str | None = None, role: bool = False) -> bytes:
        delta: dict[str, Any] = {}
        if role:
            delta["role"] = "assistant"
        if content is not None:
            delta["content"] = content
        obj = {
            "id": "x",
            "object": "chat.completion.chunk",
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(obj)}\n\n".encode()

    @app.post("/v1/chat/completions")
    async def chat(req: Request) -> StreamingResponse:
        body = await req.json()
        st.bodies.append(body)
        st.requests += 1
        n = int(body.get("max_tokens", 4))

        async def gen():
            st.inflight += 1
            st.peak_inflight = max(st.peak_inflight, st.inflight)
            try:
                yield chunk(None, role=True)
                await asyncio.sleep(ttft_s)
                for i in range(n):
                    if i:
                        await asyncio.sleep(token_delay_s)
                    st.tokens += 1
                    yield chunk(f"t{i} ")
                yield chunk(None, finish="length")
                so = body.get("stream_options") or {}
                if so.get("include_usage") and flavor != "ours":
                    usage = {
                        "prompt_tokens": len(str(body["messages"][0]["content"]).split()),
                        "completion_tokens": n,
                        "total_tokens": 0,
                        "prompt_tokens_details": {"cached_tokens": 16},
                    }
                    obj = {"id": "x", "choices": [], "usage": usage}
                    yield f"data: {json.dumps(obj)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                st.inflight -= 1

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "loop": {"healthy": True}})

    @app.get("/version")
    async def ver() -> JSONResponse:
        if flavor == "sglang":
            return JSONResponse({"detail": "not found"}, status_code=404)
        return JSONResponse({"version": version})

    @app.get("/get_server_info")
    async def info() -> JSONResponse:
        return JSONResponse({"version": version})

    @app.get("/metrics", response_model=None)
    async def metrics() -> Any:
        r = st.requests
        if flavor == "ours":
            sched: dict[str, Any] = {"step": r, "preemptions_total": r // 4}
            if not prefix_off:
                sched.update(
                    cache_evictions=r // 2, cache_blocks_reused=r * 3, cache_blocks_required=r * 4
                )
            return JSONResponse(
                {
                    "server": {"requests_received": r, "output_tokens_total": st.tokens},
                    "scheduler": sched,
                    "allocator": {"num_blocks": 2048, "block_size": 16, "tokens_capacity": 32768},
                    "registered": {"output_tok_s": 1.0},
                }
            )
        if bare:
            return PlainTextResponse("# nothing exposed\n")
        if flavor == "vllm":
            lines = [
                "# TYPE vllm:num_preemptions_total counter",
                f'vllm:num_preemptions_total{{model_name="m"}} {r // 4}',
                f'vllm:prefix_cache_hits_total{{model_name="m"}} {r * 30}',
                f'vllm:prefix_cache_queries_total{{model_name="m"}} {r * 40}',
                f'vllm:e2e_request_latency_seconds_bucket{{le="1.0",model_name="m"}} {r}',
                f'vllm:e2e_request_latency_seconds_count{{model_name="m"}} {r}',
            ]
        else:
            pe = "sglang:prefill_effective_tokens_total"
            lines = [
                "# TYPE sglang:cache_hit_rate gauge",
                'sglang:cache_hit_rate{model_name="m"} 0.99',
                f'sglang:num_retracted_requests_total{{model_name="m"}} {r // 4}',
                f'sglang:evicted_tokens_total{{model_name="m"}} {r * 7}',
                f'{pe}{{mode="device_hit",model_name="m"}} {r * 20}',
                f'{pe}{{mode="host_hit",model_name="m"}} {r * 10}',
                f'{pe}{{mode="miss",model_name="m"}} {r * 70}',
                'sglang:num_retracted_requests_created{model_name="m"} 1700000000.0',
            ]
        return PlainTextResponse("\n".join(lines) + "\n")

    return app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ThreadServer:
    """uvicorn on a background thread; `.base_url`, `.app`, `.stop()`."""

    def __init__(self, app: FastAPI) -> None:
        import uvicorn

        self.app = app
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        cfg = uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="error", lifespan="off"
        )
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("mock server did not start")
            time.sleep(0.01)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def serve_in_thread(**kw: Any) -> ThreadServer:
    return ThreadServer(make_app(**kw))


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--flavor", default="vllm")
    p.add_argument("--startup-delay", type=float, default=0.0)
    p.add_argument("--print", default="", help="line to print at startup (log-grep tests)")
    a = p.parse_args()
    if a.print:
        print(a.print, flush=True)
    time.sleep(a.startup_delay)
    uvicorn.run(make_app(flavor=a.flavor), host="127.0.0.1", port=a.port, log_level="error")


if __name__ == "__main__":
    main()
