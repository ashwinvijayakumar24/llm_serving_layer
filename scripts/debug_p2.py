"""One request through the real stack, printing every layer. Diagnostic only."""
import asyncio, json, os, sys

async def main():
    from serving.server.app import build_default_app
    import httpx
    app = build_default_app(max_batch_size=8, max_prefill_tokens=256)

    sched = None
    for attr in ("state", "extra"):
        s = getattr(app, attr, None)
        if s is not None and hasattr(s, "scheduler"):
            sched = s.scheduler
    print("scheduler found:", sched is not None)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        body = {"model": "llama-3.2-1b-instruct",
                "messages": [{"role": "user", "content": "Hello, tell me about GPUs."}],
                "max_tokens": 12, "stream": True}
        print("\n--- STREAMING ---")
        r = await c.post("/v1/chat/completions", json=body)
        print("status:", r.status_code)
        raw = r.text
        print("raw bytes:", len(raw))
        for line in raw.splitlines():
            if line.strip():
                print("  ", line[:160])

        print("\n--- NON-STREAMING ---")
        body["stream"] = False
        r2 = await c.post("/v1/chat/completions", json=body)
        print("status:", r2.status_code)
        print(json.dumps(r2.json(), indent=2)[:900])

        print("\n--- HEALTH ---")
        h = await c.get("/health")
        print(json.dumps(h.json(), indent=2)[:600])

asyncio.run(main())
