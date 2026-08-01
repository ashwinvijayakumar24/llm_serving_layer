"""
CUDA error checking and the fatal-error policy. CPU only, no GPU.

WHY THESE TESTS CAN RUN WITHOUT A GPU, AND WHY THAT MATTERS
-----------------------------------------------------------
The failure this guards against (R10) is a kernel launch that fails and is never
checked, producing garbage of the correct shape. Reproducing that on real
hardware means deliberately corrupting a CUDA context, which is not something a
test suite can do repeatably. So `CudaGuard` takes its synchronise call as an
injected callable and the error path is exercised by making that callable raise.

That is not a weaker test. The thing under test is the POLICY — that any CUDA
error is fatal, that the state latches, that in-flight requests are failed
explicitly rather than left hanging, and that /health reports it so a router
quarantines the replica. All of that is host-side logic, and a policy that could
only be tested by breaking a GPU would never be tested at all.

The end-to-end test drives the real `SchedulerLoop` with a guard whose sync
fails, and asserts on what a CLIENT and a ROUTER see: an error chunk on the open
stream, and 503 with `fatal.kind == "cuda"` from /health.

    python3 -m pytest tests/test_cuda_guard.py -q
"""

from __future__ import annotations

import asyncio

import pytest

from serving.metrics.cuda_guard import (
    CudaFatalError,
    CudaGuard,
    check_cuda_error,
    cuda_is_available,
    default_guard,
)
from serving.server.app import SchedulerLoop, ServerConfig, ServerMetrics, _Stream


def run(coro):
    """Every test owns its event loop; the scheduler task is loop-bound.

    Matches tests/test_server.py's convention rather than introducing a
    pytest-asyncio mode this suite does not otherwise configure.
    """
    return asyncio.run(coro)


def boom() -> None:
    """A simulated sticky CUDA error, in the shape torch actually raises."""
    raise RuntimeError(
        "CUDA error: an illegal memory access was encountered\n"
        "CUDA kernel errors might be asynchronously reported at some other API call"
    )


def _guard(**kw) -> CudaGuard:
    kw.setdefault("available_fn", lambda: True)
    return CudaGuard(**kw)


# ---------------------------------------------------------------------------
# check_cuda_error
# ---------------------------------------------------------------------------


def test_raises_with_the_context_name():
    with pytest.raises(CudaFatalError) as exc:
        check_cuda_error("scheduler.step", sync_fn=boom, available_fn=lambda: True)
    assert exc.value.context == "scheduler.step"
    # The raw CUDA text names no operation — "illegal memory access" is true of
    # every kernel in the process. The context is the only actionable part.
    assert "scheduler.step" in str(exc.value)
    assert "illegal memory access" in str(exc.value)


def test_error_states_the_fatal_policy():
    with pytest.raises(CudaFatalError) as exc:
        check_cuda_error("prefill", sync_fn=boom, available_fn=lambda: True)
    text = str(exc.value).lower()
    assert "fatal" in text and "poisoned" in text


def test_no_op_without_cuda():
    calls = []
    assert check_cuda_error(
        "x", sync_fn=lambda: calls.append(1), available_fn=lambda: False
    ) is False
    assert calls == [], "no synchronisation may happen when there is no CUDA"


def test_clean_sync_returns_true():
    assert check_cuda_error("x", sync_fn=lambda: None, available_fn=lambda: True) is True


def test_cuda_is_available_never_raises():
    assert isinstance(cuda_is_available(), bool)


def test_default_guard_is_a_singleton():
    assert default_guard() is default_guard()


# ---------------------------------------------------------------------------
# CudaGuard: gating and latching
# ---------------------------------------------------------------------------


def test_disabled_on_a_machine_without_cuda():
    g = CudaGuard(available_fn=lambda: False, sync_fn=boom)
    assert g.available is False and g.enabled is False
    for step in range(5):
        assert g.maybe_check("scheduler.step", step) is False
    assert g.errors_total == 0 and g.poisoned is False


def test_checks_every_step_by_default():
    n = []
    g = _guard(sync_fn=lambda: n.append(1))
    for step in range(4):
        g.maybe_check("scheduler.step", step)
    assert len(n) == 4 and g.checks_total == 4


def test_every_n_steps_gates_the_sync():
    n = []
    g = _guard(every_n_steps=3, sync_fn=lambda: n.append(1))
    for step in range(9):
        g.maybe_check("scheduler.step", step)
    assert len(n) == 3, "cost must actually drop; the knob is not decorative"
    assert g.skipped_total == 6


def test_zero_disables_checking():
    n = []
    g = _guard(every_n_steps=0, sync_fn=lambda: n.append(1))
    for step in range(5):
        assert g.maybe_check("scheduler.step", step) is False
    assert n == []


def test_negative_frequency_rejected():
    with pytest.raises(ValueError):
        CudaGuard(every_n_steps=-1)


def test_state_latches_and_the_first_error_is_the_one_reported():
    g = _guard(sync_fn=boom)
    with pytest.raises(CudaFatalError) as first:
        g.maybe_check("scheduler.step", 0)
    assert g.poisoned is True and g.errors_total == 1

    # A poisoned context that later appears to sync cleanly has not recovered.
    g.sync_fn = lambda: None
    for step in range(1, 4):
        with pytest.raises(CudaFatalError) as again:
            g.maybe_check("scheduler.step", step)
        assert again.value is first.value, "a later clean sync must not clear the fault"
    assert g.checks_total == 0, "no check after poisoning counts as a successful check"


def test_stats_report_the_fault_and_the_r2_coupling():
    g = _guard(sync_fn=boom)
    with pytest.raises(CudaFatalError):
        g.check("scheduler.step")
    s = g.stats()
    assert s["poisoned"] is True
    assert s["errors_total"] == 1
    assert s["context"] == "scheduler.step"
    assert "illegal memory access" in s["error"]
    # The sync is not purely a cost: it is also what keeps host-clock step timing
    # a measure of execution rather than launch queueing (R2).
    assert "queueing" in s["sync_note"]


# ---------------------------------------------------------------------------
# The policy, end to end through SchedulerLoop
# ---------------------------------------------------------------------------


class OkScheduler:
    """Steps forever without complaint. The CUDA error is the only fault."""

    def __init__(self):
        self.steps = 0
        self.has_work = True

    def step(self):
        self.steps += 1

    def snapshot(self):
        return {"step": self.steps, "running": 1, "waiting": 0,
                "blocks_free": 1, "blocks_used": 1, "block_utilization": 0.5}


async def _drive(loop: SchedulerLoop, deadline_s: float = 1.0) -> None:
    loop.ensure_started()
    t0 = asyncio.get_running_loop().time()
    while loop.running and asyncio.get_running_loop().time() - t0 < deadline_s:
        await asyncio.sleep(0.001)


async def _test_cuda_error_stops_the_loop_and_marks_it_unhealthy():
    sched = OkScheduler()
    loop = SchedulerLoop(sched, ServerMetrics(), cuda_guard=_guard(sync_fn=boom))
    await _drive(loop)

    assert loop.running is False, "a poisoned replica must not keep stepping"
    assert loop.healthy is False
    assert loop.fatal_kind == "cuda"
    assert "illegal memory access" in (loop.last_error or "")
    assert sched.steps <= 2, "detection must be within one step, not eventual"


async def _test_in_flight_requests_are_failed_explicitly_not_left_hanging():
    """
    The difference between a fault a client can see and a fault a client
    attributes to latency. A replica that merely stops stepping leaves every open
    SSE stream hanging until the client's timeout, and a timeout is charged to the
    latency distribution rather than to a failure count.
    """
    loop = SchedulerLoop(OkScheduler(), ServerMetrics(), cuda_guard=_guard(sync_fn=boom))
    streams = {f"req-{i}": _Stream() for i in range(3)}
    for rid, st in streams.items():
        loop.register(rid, st)

    await _drive(loop)

    for rid, st in streams.items():
        assert st.finished, f"{rid} was left hanging"
        kind, payload = st.queue.get_nowait()
        assert kind == "error"
        assert "CUDA" in payload and "scheduler.step" in payload


async def _test_health_reports_the_fault_so_a_router_can_quarantine():
    loop = SchedulerLoop(OkScheduler(), ServerMetrics(), cuda_guard=_guard(sync_fn=boom))
    await _drive(loop)

    stats = loop.stats()
    assert stats["healthy"] is False
    assert stats["fatal_kind"] == "cuda"
    assert stats["cuda"]["poisoned"] is True
    assert stats["cuda"]["errors_total"] == 1


async def _test_a_healthy_loop_with_a_working_sync_keeps_running():
    """The guard must not be a liveness hazard: a clean sync changes nothing."""
    loop = SchedulerLoop(OkScheduler(), ServerMetrics(),
                         cuda_guard=_guard(sync_fn=lambda: None))
    loop.ensure_started()
    await asyncio.sleep(0.05)
    assert loop.running and loop.healthy and loop.fatal_kind is None
    assert loop.steps > 0
    assert loop.cuda_guard.checks_total == loop.steps or \
           loop.cuda_guard.checks_total == loop.steps + 1
    await loop.aclose()


async def _test_ordinary_step_failure_is_labelled_differently():
    """A CUDA fault and a Python bug are both fatal, but a router should be able
    to tell a poisoned context from a code error when deciding whether a restart
    can possibly help."""

    class Exploding(OkScheduler):
        def step(self):
            raise ValueError("bad batch")

    loop = SchedulerLoop(Exploding(), ServerMetrics(),
                         cuda_guard=_guard(sync_fn=lambda: None))
    await _drive(loop)
    assert loop.healthy is False
    assert loop.fatal_kind == "step"
    assert "ValueError" in (loop.last_error or "")


async def _test_server_health_endpoint_reports_cuda_fatal_and_503():
    import httpx

    from serving.server.app import create_app

    app = create_app(OkScheduler(), _NullTokenizer(), config=ServerConfig())
    # Swap the guard on the already-built loop: create_app builds one from the
    # config, and there is no CUDA here to fail on its own.
    app.state.scheduler_loop.cuda_guard = _guard(sync_fn=boom)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        first = await client.get("/health")
        assert first.status_code in (200, 503)
        for _ in range(50):
            resp = await client.get("/health")
            if resp.status_code == 503:
                break
            await asyncio.sleep(0.005)

    body = resp.json()
    assert resp.status_code == 503
    assert body["status"] == "unhealthy"
    assert body["fatal"]["kind"] == "cuda"
    assert body["fatal"]["restart_required"] is True
    assert body["fatal"]["quarantine"] is True


class _NullTokenizer:
    def encode_chat(self, messages):
        return [1, 2, 3]

    def decode(self, token_ids):
        return ""


# ---------------------------------------------------------------------------
# Sync entry points. The coroutines above are driven through `run` so each test
# owns its event loop, matching tests/test_server.py.
# ---------------------------------------------------------------------------


def test_cuda_error_stops_the_loop_and_marks_it_unhealthy():
    run(_test_cuda_error_stops_the_loop_and_marks_it_unhealthy())


def test_in_flight_requests_are_failed_explicitly_not_left_hanging():
    run(_test_in_flight_requests_are_failed_explicitly_not_left_hanging())


def test_health_reports_the_fault_so_a_router_can_quarantine():
    run(_test_health_reports_the_fault_so_a_router_can_quarantine())


def test_a_healthy_loop_with_a_working_sync_keeps_running():
    run(_test_a_healthy_loop_with_a_working_sync_keeps_running())


def test_ordinary_step_failure_is_labelled_differently():
    run(_test_ordinary_step_failure_is_labelled_differently())


def test_server_health_endpoint_reports_cuda_fatal_and_503():
    run(_test_server_health_endpoint_reports_cuda_fatal_and_503())
