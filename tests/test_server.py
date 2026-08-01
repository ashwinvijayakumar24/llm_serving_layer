"""
HTTP surface tests. CPU only, no GPU, no weights, no network.

Everything here runs against `serving.server.app.create_app` wired to the REAL
`serving.scheduler.scheduler.Scheduler` and the REAL `BlockAllocator`, with only
the model and the tokenizer faked. That combination is deliberate: faking the
scheduler too would make the concurrency test a test of the fake.

THE LOAD-BEARING TEST IS `test_two_concurrent_streams_interleave`.
It is the direct regression test for `engine/server.py:69`, which iterates a
blocking synchronous generator inside an `async def` and therefore pins the
event loop for a whole generation — making request 2's TTFT include request 1's
entire decode. Everything else in this file protects a response shape; that one
protects whether "concurrent" means anything here at all.

HOW CONCURRENCY IS OBSERVED
---------------------------
Not through `httpx`. `httpx.ASGITransport` accumulates the whole response body
before returning it (`httpx/_transports/asgi.py`, `ASGIResponseStream`), so a
client-side chunk timing through it would be an artefact of the transport rather
than a measurement of the server. `SSEDriver` below therefore speaks ASGI to the
app directly and records every `http.response.body` message, from both requests,
into ONE ordered log. Interleaving is then a statement about that log's order,
observed at the transport boundary with nothing mocked in between.

    python3 -m pytest tests/test_server.py -q
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest
import torch

from bench.loadgen import LoadGenConfig, Outcome, Phase, RequestSpec, stream_one
from serving.memory.allocator import BlockAllocator
from serving.scheduler.scheduler import EOS_IDS, RequestState, Scheduler, SchedulerConfig
from serving.server.app import ServerConfig, create_app

VOCAB = 128_016
EOS = 128_009
assert EOS in EOS_IDS

#: Detokenizes to "" — a special token, i.e. exactly the case
#: docs/BENCHMARK_METHODOLOGY.md §2 calls a definitional trap.
SPECIAL_ID = 100

#: Last prompt token. The fake model's chain is `v -> v + 1`, so the first
#: generated token is SPECIAL_ID and every stream starts with an invisible one.
PROMPT_TAIL = SPECIAL_ID - 1


# ---------------------------------------------------------------------------
# Fakes — the model and the tokenizer, and nothing else
# ---------------------------------------------------------------------------


class FakeModel:
    """
    Deterministic stand-in for `LlamaModelGPU`.

    Contract with the scheduler is exactly `forward_varlen(tokens, meta, backend)
    -> (n_seqs, vocab)` on `self.device`. The next token is `last_input_token + 1`,
    which makes each sequence a strictly increasing chain the test can predict
    without knowing anything about batch composition — the point being that the
    scheduler may batch these sequences however it likes and the expected output
    does not change.
    """

    def __init__(self, eos_at: int | None = None):
        self.device = torch.device("cpu")
        self.eos_at = eos_at
        self.calls = 0
        self.tokens_seen = 0

    def forward_varlen(self, tokens: torch.Tensor, meta: Any, backend: Any) -> torch.Tensor:
        self.calls += 1
        self.tokens_seen += int(tokens.numel())
        last_ix = meta.last_token_ix.tolist()
        logits = torch.full((len(last_ix), VOCAB), -1e4, dtype=torch.float32)
        for row, ix in enumerate(last_ix):
            prev = int(tokens[ix].item())
            nxt = EOS if (self.eos_at is not None and prev >= self.eos_at) else prev + 1
            logits[row, nxt] = 1.0
        return logits


class FakeTokenizer:
    """
    Ids in, text out. `SPECIAL_ID` and the EOS ids render as "", matching what a
    real tokenizer does under `skip_special_tokens=True`.
    """

    def __init__(self, prompt_len: int = 3):
        self.prompt_len = prompt_len
        self.encoded: list[list[dict[str, str]]] = []

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        self.encoded.append(messages)
        return [10] * (self.prompt_len - 1) + [PROMPT_TAIL]

    def decode(self, token_ids) -> str:
        return "".join(
            "" if t in EOS_IDS or t == SPECIAL_ID else f"t{t}" for t in token_ids
        )


def expected_text(max_tokens: int, eos_at: int | None = None) -> str:
    """The visible content a stream of `max_tokens` tokens must produce."""
    out, v = [], PROMPT_TAIL
    for _ in range(max_tokens):
        v = EOS if (eos_at is not None and v >= eos_at) else v + 1
        out.append(v)
        if v in EOS_IDS:
            break
    return FakeTokenizer().decode(out)


class SpyScheduler:
    """
    Transparent proxy over the real Scheduler that records `cancel` calls.

    A spy rather than a mock: every call still reaches the real scheduler, so the
    disconnect tests assert on the recorded id AND on the allocator actually
    getting its blocks back.
    """

    def __init__(self, inner: Scheduler):
        self.inner = inner
        self.cancelled: list[str] = []

    def cancel(self, request_id: str) -> bool:
        self.cancelled.append(request_id)
        return self.inner.cancel(request_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class ExplodingScheduler:
    """A scheduler whose step raises. Used for the loop-death / 503 path."""

    def __init__(self, inner: Scheduler):
        self.inner = inner
        self.allocator = inner.allocator

    def step(self):
        raise RuntimeError("simulated CUDA fault")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------


def make_app(
    *,
    eos_at: int | None = None,
    max_waiting: int = 1024,
    max_batch_size: int = 8,
    max_prefill_tokens: int = 64,
    num_blocks: int = 512,
    spy: bool = False,
    exploding: bool = False,
    disconnect_poll_s: float = 0.01,
):
    model = FakeModel(eos_at=eos_at)
    allocator = BlockAllocator(num_blocks=num_blocks, block_size=16, watermark_blocks=8)
    inner = Scheduler(
        model, backend=None, allocator=allocator,
        config=SchedulerConfig(
            max_batch_size=max_batch_size,
            max_prefill_tokens=max_prefill_tokens,
            max_waiting=max_waiting,
        ),
    )
    scheduler: Any = inner
    if spy:
        scheduler = SpyScheduler(inner)
    if exploding:
        scheduler = ExplodingScheduler(inner)
    tok = FakeTokenizer()
    app = create_app(
        scheduler, tok,
        config=ServerConfig(idle_sleep_s=0.001, disconnect_poll_s=disconnect_poll_s),
    )
    return app, scheduler, inner, tok


def body(max_tokens: int = 8, stream: bool = True, prompt: str = "hello") -> dict:
    return {
        "model": "fake",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.0,
    }


def run(coro):
    """Every test owns its event loop; the app's scheduler task is loop-bound."""
    return asyncio.run(coro)


async def shutdown(app) -> None:
    await app.state.scheduler_loop.aclose()


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


# ---------------------------------------------------------------------------
# A minimal ASGI driver — the only way to observe chunk ORDER honestly
# ---------------------------------------------------------------------------


class SSEDriver:
    """
    Speaks ASGI to the app and records every response-body message.

    `log` is SHARED across drivers, so the concatenation of two concurrent
    requests' `send` calls is a single ordered transcript of what the server
    actually emitted, in the order it emitted it.

    `disconnect` lets a test deliver `http.disconnect` at a chosen moment, which
    is how the cancellation path is exercised without a socket.
    """

    def __init__(self, app, payload: dict, tag: str, log: list, path: str = "/v1/chat/completions"):
        self.app = app
        self.payload = payload
        self.tag = tag
        self.log = log
        self.path = path
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self.chunks: list[bytes] = []
        self.disconnect = asyncio.Event()
        self.content_gate: tuple[int, asyncio.Event] | None = None
        self._content_seen = 0

    # -- transcript views ---------------------------------------------------

    @property
    def raw(self) -> bytes:
        return b"".join(self.chunks)

    def events(self) -> list[Any]:
        """Parsed `data:` payloads, `'[DONE]'` kept as a literal string."""
        out = []
        for line in self.raw.decode().splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            p = line[len("data:"):].strip()
            out.append("[DONE]" if p == "[DONE]" else json.loads(p))
        return out

    def contents(self) -> list[str]:
        return [
            c for e in self.events()
            if isinstance(e, dict)
            for c in [e["choices"][0]["delta"].get("content", "")]
            if c
        ]

    # -- ASGI ---------------------------------------------------------------

    async def run(self) -> None:
        raw_body = json.dumps(self.payload).encode()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": self.path,
            "raw_path": self.path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"test"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(raw_body)).encode()),
            ],
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
        }
        body_sent = False

        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": raw_body, "more_body": False}
            await self.disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                self.status = message["status"]
                self.headers = {
                    k.decode(): v.decode() for k, v in message.get("headers", [])
                }
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                if not chunk:
                    return
                self.chunks.append(chunk)
                self.log.append((time.perf_counter(), self.tag, chunk))
                if b'"content"' in chunk:
                    self._content_seen += 1
                    if self.content_gate and self._content_seen >= self.content_gate[0]:
                        self.content_gate[1].set()

        await self.app(scope, receive, send)


def is_content(chunk: bytes) -> bool:
    obj = json.loads(chunk.decode().split("data:", 1)[1].strip())
    return bool(obj["choices"][0]["delta"].get("content"))


def is_terminal(chunk: bytes) -> bool:
    payload = chunk.decode().split("data:", 1)[1].strip()
    if payload == "[DONE]":
        return False
    return json.loads(payload)["choices"][0].get("finish_reason") is not None


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


def test_non_streaming_returns_openai_shape_with_usage():
    app, *_ = make_app()

    async def go():
        async with client_for(app) as c:
            r = await c.post("/v1/chat/completions", json=body(max_tokens=6, stream=False))
        await shutdown(app)
        return r

    r = run(go())
    assert r.status_code == 200
    d = r.json()
    assert d["object"] == "chat.completion"
    assert d["id"].startswith("chatcmpl-")
    assert isinstance(d["created"], int)
    choice = d["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == expected_text(6)
    assert choice["finish_reason"] == "length"
    assert d["usage"] == {
        "prompt_tokens": 3,
        # 6 sampled tokens; one of them is SPECIAL_ID, which is invisible in the
        # TEXT but is still a generated token and is still counted here.
        "completion_tokens": 6,
        "total_tokens": 9,
    }


def test_non_streaming_eos_reports_finish_reason_stop():
    app, *_ = make_app(eos_at=PROMPT_TAIL + 4)

    async def go():
        async with client_for(app) as c:
            r = await c.post("/v1/chat/completions", json=body(max_tokens=64, stream=False))
        await shutdown(app)
        return r

    d = run(go()).json()
    assert d["choices"][0]["finish_reason"] == "stop"
    # The EOS token itself is a control signal, never content.
    assert f"t{EOS}" not in d["choices"][0]["message"]["content"]
    assert d["choices"][0]["message"]["content"] == expected_text(64, eos_at=PROMPT_TAIL + 4)
    assert d["usage"]["completion_tokens"] < 64


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_streaming_emits_chunks_then_terminal_then_done():
    app, *_ = make_app()
    log: list = []
    d = SSEDriver(app, body(max_tokens=5), "r1", log)

    async def go():
        await d.run()
        await shutdown(app)

    run(go())

    assert d.status == 200
    assert d.headers["content-type"].startswith("text/event-stream")
    events = d.events()

    assert events[-1] == "[DONE]"
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    for e in events[:-1]:
        assert e["object"] == "chat.completion.chunk"
        assert e["choices"][0]["index"] == 0

    terminal = events[-2]
    assert terminal["choices"][0]["delta"] == {}
    assert terminal["choices"][0]["finish_reason"] == "length"
    # Exactly one finish_reason in the whole stream, and it is the last event
    # before [DONE]. A finish_reason on an intermediate chunk would make a client
    # stop reading early.
    assert sum(
        1 for e in events[:-1] if e["choices"][0]["finish_reason"] is not None
    ) == 1

    assert "".join(d.contents()) == expected_text(5)
    assert all(e["id"] == events[0]["id"] for e in events[:-1])


def test_streaming_eos_terminates_with_stop():
    app, *_ = make_app(eos_at=PROMPT_TAIL + 3)
    log: list = []
    d = SSEDriver(app, body(max_tokens=200), "r1", log)

    async def go():
        await d.run()
        await shutdown(app)

    run(go())
    events = d.events()
    assert events[-1] == "[DONE]"
    assert events[-2]["choices"][0]["finish_reason"] == "stop"
    assert "".join(d.contents()) == expected_text(200, eos_at=PROMPT_TAIL + 3)


# ---------------------------------------------------------------------------
# THE test — concurrency
# ---------------------------------------------------------------------------


def test_two_concurrent_streams_interleave():
    """
    Two simultaneous streaming requests must INTERLEAVE, not serialize.

    This is the regression test for `engine/server.py:69`. Under that design the
    event loop is pinned by request 1's synchronous decode generator, so request
    2 cannot be admitted, cannot be scheduled, and cannot emit a byte until
    request 1 has run to completion — its first token would land AFTER request
    1's terminal chunk in this transcript.

    The gate makes the test deterministic rather than timing-dependent: request 2
    is not started until request 1 has already emitted five content chunks. If
    the loop were pinned, that gate could never even fire until request 1 was
    done, and request 2's first token would necessarily follow request 1's last.
    """
    app, _, inner, _ = make_app(max_batch_size=8)
    log: list = []
    gate = asyncio.Event()

    d1 = SSEDriver(app, body(max_tokens=200), "r1", log)
    d2 = SSEDriver(app, body(max_tokens=200), "r2", log)
    d1.content_gate = (5, gate)

    started: dict[str, float] = {}

    async def second():
        await asyncio.wait_for(gate.wait(), timeout=10)
        started["r2"] = time.perf_counter()
        await d2.run()

    async def go():
        await asyncio.wait_for(asyncio.gather(d1.run(), second()), timeout=30)
        await shutdown(app)

    run(go())

    assert d1.status == 200 and d2.status == 200

    entries = [(t, tag, ch) for t, tag, ch in log if ch.startswith(b"data: {")]
    r2_first_token = next(
        i for i, (_, tag, ch) in enumerate(entries) if tag == "r2" and is_content(ch)
    )
    r1_terminal = next(
        i for i, (_, tag, ch) in enumerate(entries) if tag == "r1" and is_terminal(ch)
    )

    # ------------------------------------------------------------------
    # THE ASSERTION, in the form the bug is stated in: request 2's TTFT must
    # not contain request 1's decode.
    #
    # Request 2 was dispatched after request 1 had emitted 5 of its 200 tokens,
    # so if the loop were pinned its first token could not appear until request 1
    # had run out the remaining 195 — i.e. its TTFT would be most of request 1's
    # whole stream. Measured against that span rather than an absolute
    # millisecond budget, because the budget would be a statement about this
    # machine and the ratio is a statement about the server.
    # ------------------------------------------------------------------
    ttft2 = entries[r2_first_token][0] - started["r2"]
    r1_span = entries[r1_terminal][0] - entries[0][0]
    assert ttft2 < 0.25 * r1_span, (
        f"request 2's TTFT was {ttft2 * 1e3:.1f}ms against request 1's "
        f"{r1_span * 1e3:.1f}ms stream — request 2 waited out request 1's decode, "
        "which is exactly the engine's event-loop-pinning bug (engine/server.py:69)"
    )

    # Ordering, stated directly: request 2 was producing tokens before request 1
    # had finished, and not by one chunk on a lucky race — request 1 still had
    # the large majority of its 200 tokens left to emit.
    assert r2_first_token < r1_terminal
    r1_after = sum(
        1 for _, tag, ch in entries[r2_first_token:] if tag == "r1" and is_content(ch)
    )
    assert r1_after >= 150, f"only {r1_after} of r1's tokens overlapped r2"

    # Interleaved at the granularity of a scheduler step: while both sequences
    # are in the running batch every step emits one token for each, so neither
    # request ever gets a long uninterrupted run. A serialized server produces
    # one run of ~200 followed by another.
    overlap = [tag for _, tag, ch in entries[r2_first_token:r1_terminal] if is_content(ch)]
    assert len(set(overlap)) == 2
    longest_run, cur = 1, 1
    for a, b in zip(overlap, overlap[1:], strict=False):
        cur = cur + 1 if a == b else 1
        longest_run = max(longest_run, cur)
    assert longest_run <= 4, f"longest single-request run inside the overlap was {longest_run}"

    # Both streams are complete and correct despite having been batched together
    # — batching must not change output (the batch-invariance property).
    assert "".join(d1.contents()) == expected_text(200)
    assert "".join(d2.contents()) == expected_text(200)

    # Every block came back.
    inner.allocator.check_invariants()
    assert inner.allocator.num_used == 0


def test_health_stays_responsive_while_a_long_stream_decodes():
    """
    A second, independent proof that the loop is not pinned: an unrelated HTTP
    endpoint answers while a 500-token generation is mid-flight. Under the
    engine's design this GET would block until the stream finished.
    """
    app, _, inner, _ = make_app()
    log: list = []
    gate = asyncio.Event()
    d = SSEDriver(app, body(max_tokens=500), "r1", log)
    d.content_gate = (5, gate)
    seen: dict[str, Any] = {}

    async def probe():
        await asyncio.wait_for(gate.wait(), timeout=10)
        async with client_for(app) as c:
            r = await c.get("/health")
        seen["status"] = r.status_code
        seen["snapshot"] = r.json()["scheduler"]

    async def go():
        await asyncio.wait_for(asyncio.gather(d.run(), probe()), timeout=30)
        await shutdown(app)

    run(go())
    assert seen["status"] == 200
    # Answered WHILE the request was running, not after it retired.
    assert seen["snapshot"]["running"] == 1
    assert seen["snapshot"]["blocks_used"] > 0
    assert len(d.contents()) == 499  # 500 tokens, one of them invisible


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_client_disconnect_cancels_the_request_and_frees_blocks():
    """
    ASGI `http.disconnect` mid-stream must reach `scheduler.cancel(request_id)`.

    This is the leak guard. A disconnect that does not cancel leaves the request
    generating to `max_tokens` while holding its blocks; over a disconnect-heavy
    workload that drains a FINITE SHARED pool, and the symptom appears as
    admission failures on unrelated requests with no error anywhere.
    """
    app, spy, inner, _ = make_app(spy=True)
    log: list = []
    gate = asyncio.Event()
    d = SSEDriver(app, body(max_tokens=5000), "r1", log)
    d.content_gate = (3, gate)

    async def go():
        task = asyncio.create_task(d.run())
        await asyncio.wait_for(gate.wait(), timeout=10)
        blocks_held = inner.allocator.num_used
        d.disconnect.set()                       # the client goes away
        await asyncio.wait_for(task, timeout=10)
        # Cancellation only MARKS; the blocks come back at the next step
        # boundary, because freeing them inline could pull memory out from
        # under a forward pass that is mid-flight.
        for _ in range(50):
            await asyncio.sleep(0.005)
            if inner.allocator.num_used == 0:
                break
        await shutdown(app)
        return blocks_held

    blocks_held = run(go())

    assert blocks_held > 0
    assert len(spy.cancelled) == 1
    request_id = spy.cancelled[0]
    assert request_id.startswith("chatcmpl-")
    # The id cancelled is the id of the request that was streaming, not some
    # other in-flight request.
    assert request_id == d.events()[0]["id"]

    cancelled = [r for r in inner.finished if r.request_id == request_id]
    assert len(cancelled) == 1
    assert cancelled[0].state == RequestState.CANCELLED
    assert len(cancelled[0].output_ids) < 5000, "generation continued after disconnect"

    inner.allocator.check_invariants()
    assert inner.allocator.num_used == 0, "disconnected request leaked its blocks"


def test_task_cancellation_path_also_cancels():
    """
    The OTHER teardown mechanism: the response task is cancelled outright, so the
    SSE generator sees `CancelledError`/`GeneratorExit` at its `yield` rather
    than an `http.disconnect` message. Which of the two fires depends on the
    server, so both are handled and both are tested.
    """
    app, spy, inner, _ = make_app(spy=True)
    log: list = []
    gate = asyncio.Event()
    d = SSEDriver(app, body(max_tokens=5000), "r1", log)
    d.content_gate = (3, gate)

    async def go():
        task = asyncio.create_task(d.run())
        await asyncio.wait_for(gate.wait(), timeout=10)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        for _ in range(50):
            await asyncio.sleep(0.005)
            if inner.allocator.num_used == 0:
                break
        await shutdown(app)

    run(go())
    assert len(spy.cancelled) == 1
    inner.allocator.check_invariants()
    assert inner.allocator.num_used == 0


def test_non_streaming_disconnect_also_cancels():
    """A non-streaming client that leaves leaks just as effectively; it just has
    no chunk boundary at which anyone would notice."""
    app, spy, inner, _ = make_app(spy=True)
    log: list = []
    d = SSEDriver(app, body(max_tokens=5000, stream=False), "r1", log)

    async def go():
        task = asyncio.create_task(d.run())
        while inner.allocator.num_used == 0:
            await asyncio.sleep(0.002)
        d.disconnect.set()
        await asyncio.wait_for(task, timeout=10)
        for _ in range(100):
            await asyncio.sleep(0.005)
            if inner.allocator.num_used == 0:
                break
        await shutdown(app)

    run(go())
    assert len(spy.cancelled) == 1
    assert inner.allocator.num_used == 0


# ---------------------------------------------------------------------------
# Load shedding
# ---------------------------------------------------------------------------


def test_queue_full_returns_429_and_does_not_queue():
    """
    `Scheduler.add_request` returning False is a SHED, not a retry-later queue
    entry. Queueing without bound under overload turns a throughput problem into
    an unbounded latency problem in which every request misses its SLO.
    """
    app, _, inner, _ = make_app(max_waiting=0)

    async def go():
        async with client_for(app) as c:
            r = await c.post("/v1/chat/completions", json=body(stream=True))
            m = await c.get("/metrics")
        await shutdown(app)
        return r, m

    r, m = run(go())
    assert r.status_code == 429
    assert r.headers.get("retry-after") == "1"
    err = r.json()["error"]
    assert err["type"] == "queue_full"
    assert "queue full" in err["message"]
    assert err["scheduler"]["waiting"] == 0

    # Shed means SHED: nothing was queued and no blocks were taken.
    assert inner.waiting == [] and inner.running == []
    assert inner.allocator.num_used == 0
    assert m.json()["server"]["requests_shed_429"] == 1
    assert m.json()["server"]["requests_completed"] == 0


# ---------------------------------------------------------------------------
# /health and /metrics
# ---------------------------------------------------------------------------


def test_health_shape():
    app, *_ = make_app()

    async def go():
        async with client_for(app) as c:
            r = await c.get("/health")
        await shutdown(app)
        return r

    r = run(go())
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["uptime_s"] > 0
    snap = d["scheduler"]
    assert set(snap) >= {
        "running", "waiting", "finished", "blocks_free", "blocks_used", "block_utilization"
    }
    assert snap["blocks_free"] == 512 and snap["blocks_used"] == 0
    assert 0.0 <= snap["block_utilization"] <= 1.0
    assert d["loop"]["running"] is True and d["loop"]["healthy"] is True
    # Greedy-only is DECLARED, not left to be discovered by a client that passes
    # temperature=0.7 and gets greedy output back with no indication.
    assert "greedy" in d["capabilities"]["sampling"]


def test_health_reports_503_and_streams_fail_when_the_scheduler_loop_dies():
    app, sched, inner, _ = make_app(exploding=True)
    log: list = []
    d = SSEDriver(app, body(max_tokens=10), "r1", log)
    seen: dict[str, Any] = {}

    async def go():
        await asyncio.wait_for(d.run(), timeout=10)
        async with client_for(app) as c:
            r = await c.get("/health")
        seen["status"] = r.status_code
        seen["body"] = r.json()
        await shutdown(app)

    run(go())

    # The in-flight stream is terminated explicitly rather than hanging forever.
    events = d.events()
    assert events[-1] == "[DONE]"
    assert events[-2]["choices"][0]["finish_reason"] == "error"

    assert seen["status"] == 503
    assert seen["body"]["status"] == "unhealthy"
    assert "simulated CUDA fault" in seen["body"]["loop"]["last_error"]


def test_metrics_shape_and_registered_names():
    app, *_ = make_app()

    async def go():
        async with client_for(app) as c:
            await c.post("/v1/chat/completions", json=body(max_tokens=4, stream=False))
            r = await c.get("/metrics")
        await shutdown(app)
        return r

    d = run(go()).json()

    # Registered names mean what serving/metrics/artifact.py says they mean, and
    # the spec travels with the number so a reader never has to guess (R16).
    assert "output_tok_s" in d["registered"]
    assert d["registered"]["output_tok_s"] >= 0
    spec = d["metric_specs"]["output_tok_s"]
    assert spec["unit"] == "tokens/s" and spec["source"] == "server aggregate"
    for name in d["registered"]:
        assert name in d["metric_specs"]

    # queue_wait_ms is registered but deliberately absent: the server cannot see
    # intended dispatch, and emitting a different quantity under that name is
    # precisely the engine's peak_mem_mb mistake.
    assert "queue_wait_ms" not in d["registered"]
    assert "queue_wait_ms" in d["not_emitted"]

    s = d["server"]
    assert s["requests_received"] == 1
    assert s["requests_completed"] == 1
    assert s["output_tokens_total"] == 4
    assert s["prompt_tokens_total"] == 3
    assert s["requests_shed_429"] == 0

    a = d["allocator"]
    assert a["num_blocks"] == 512 and a["block_size"] == 16 and a["watermark_blocks"] == 8
    assert a["blocks_free"] == 512 and a["blocks_used"] == 0
    assert a["total_allocated"] == a["total_freed"] > 0
    assert a["tokens_capacity"] == 512 * 16

    assert d["loop"]["steps_total"] > 0
    # Host-clock step timing is published as diagnostic, WITH the R2 caveat
    # attached to the number rather than living only in a doc.
    assert "R2" in d["loop"]["step_duration_host_ms"]["sync_note"]
    assert set(d["scheduler"]) >= {"running", "waiting", "blocks_free", "blocks_used"}


# ---------------------------------------------------------------------------
# Interop with the real benchmark client
# ---------------------------------------------------------------------------


def test_stream_is_parseable_by_the_real_loadgen_client():
    """
    Feed this server's actual SSE output to `bench.loadgen.stream_one` — the real
    parser, imported, not reimplemented. If the harness cannot read this server,
    every number the project publishes is zero.
    """
    app, *_ = make_app()
    cfg = LoadGenConfig(url="http://test/v1/chat/completions", request_timeout_s=30)
    spec = RequestSpec(
        request_id=0, intended_send_time=0.0, prompt="hello", max_tokens=7, phase=Phase.STEADY
    )

    async def go():
        async with client_for(app) as c:
            res = await stream_one(c, spec, cfg, t0=time.perf_counter())
        await shutdown(app)
        return res

    res = run(go())

    assert res.outcome == Outcome.COMPLETED, res.error
    assert res.status_code == 200
    assert res.finish_reason == "length"
    # 7 sampled tokens; the first is a special token that renders empty and is
    # therefore not a token for timing purposes (BENCHMARK_METHODOLOGY §2).
    assert res.output_tokens == 6
    assert res.text_chars == len(expected_text(7))
    assert res.ttft_ms is not None and res.e2e_ms is not None
    assert len(res.itls_ms) == 5


def test_empty_detokenizations_never_appear_as_visible_content():
    """
    Tokens that detokenize to "" must not be emitted as content chunks.

    `engine/server.py:70-78` emits one chunk per token id and detokenizes with
    `skip_special_tokens=True`, so a leading special token produces a chunk whose
    `content` is "". Any client timing TTFT to the first CHUNK then understates
    it by however many special tokens the chat template happens to start with.
    The load generator defends itself by timing to the first NON-EMPTY content
    (`RequestResult.ttft_ms`); this server removes the trap at the source and
    counts the suppressions so they cannot go unnoticed.
    """
    app, *_ = make_app()
    cfg = LoadGenConfig(url="http://test/v1/chat/completions", request_timeout_s=30)
    spec = RequestSpec(
        request_id=0, intended_send_time=0.0, prompt="hello", max_tokens=4, phase=Phase.STEADY
    )
    log: list = []
    d = SSEDriver(app, body(max_tokens=4), "r1", log)

    async def go():
        await d.run()
        async with client_for(app) as c:
            res = await stream_one(c, spec, cfg, t0=time.perf_counter())
            m = await c.get("/metrics")
        await shutdown(app)
        return res, m.json()

    res, metrics = run(go())

    events = d.events()
    # No chunk anywhere carries content == "".
    for e in events[:-1]:
        delta = e["choices"][0]["delta"]
        assert "content" not in delta or delta["content"] != ""

    # The only chunks without content are the OpenAI role chunk and the terminal
    # chunk — both structural, neither a token.
    empty = [e for e in events[:-1] if not e["choices"][0]["delta"].get("content")]
    assert len(empty) == 2
    assert empty[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert empty[1]["choices"][0]["finish_reason"] == "length"

    # The load generator agrees, and its TTFT is anchored to real text.
    assert res.empty_chunks == 2
    assert res.output_tokens == 3
    assert res.ttft_ms is not None

    # Suppressed, not hidden.
    assert metrics["server"]["empty_detokenizations_total"] >= 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": [], "max_tokens": 4},
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 0},
        {"max_tokens": 4},
    ],
)
def test_malformed_bodies_are_rejected(payload):
    app, *_ = make_app()

    async def go():
        async with client_for(app) as c:
            r = await c.post("/v1/chat/completions", json=payload)
        await shutdown(app)
        return r

    assert run(go()).status_code == 422


def test_max_tokens_is_capped_by_server_config():
    """A client asking for 10^6 tokens does not get to hold blocks for 10^6 tokens."""
    model = FakeModel()
    allocator = BlockAllocator(num_blocks=512, block_size=16, watermark_blocks=8)
    sched = Scheduler(model, None, allocator, SchedulerConfig(max_prefill_tokens=64))
    app = create_app(
        sched, FakeTokenizer(),
        config=ServerConfig(idle_sleep_s=0.001, disconnect_poll_s=0.01, max_tokens_cap=5),
    )

    async def go():
        async with client_for(app) as c:
            r = await c.post("/v1/chat/completions", json=body(max_tokens=1_000_000, stream=False))
        await shutdown(app)
        return r

    d = run(go()).json()
    assert d["usage"]["completion_tokens"] == 5
    assert d["choices"][0]["finish_reason"] == "length"


def test_oversized_prompt_is_rejected_with_413():
    model = FakeModel()
    allocator = BlockAllocator(num_blocks=512, block_size=16, watermark_blocks=8)
    sched = Scheduler(model, None, allocator, SchedulerConfig(max_prefill_tokens=64))
    app = create_app(
        sched, FakeTokenizer(prompt_len=50),
        config=ServerConfig(idle_sleep_s=0.001, max_prompt_tokens=10),
    )

    async def go():
        async with client_for(app) as c:
            r = await c.post("/v1/chat/completions", json=body(stream=False))
        await shutdown(app)
        return r

    r = run(go())
    assert r.status_code == 413
    assert "50 tokens" in r.json()["error"]["message"]
    assert "cap is 10" in r.json()["error"]["message"]
