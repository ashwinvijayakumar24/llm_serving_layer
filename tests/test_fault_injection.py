"""
Fault-injection accounting. CPU only, no GPU, no real network, nothing killed.

WHAT IS ACTUALLY UNDER TEST
---------------------------
Not "does the harness send requests" — that is httpx's job. The thing under test
is the ACCOUNTING IDENTITY, because a failover bug does not stop a run: it
silently loses the requests that were in flight on the replica that died, and a
throughput chart drawn over the survivors looks entirely normal. The property
that catches it is that every issued request must have exactly one terminal
disposition, and that a request without one is an ERROR rather than a rounding
difference.

So the tests below deliberately include the case a summarise-the-responses
harness cannot represent: a record that never reached a terminal state. If
`account()` can be made to report a clean run over a set containing one of those,
the harness cannot back the Phase 5 failover claim.

The end-to-end path is exercised against an in-process ASGI app through
`httpx.MockTransport`, so there is no server, no port, and no GPU — the kill is
a flag flip that makes the fake target start refusing connections.

    python3 -m pytest tests/test_fault_injection.py -q
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from bench.fault_injection import (
    Accounting,
    AccountingError,
    Attempt,
    Disposition,
    FaultConfig,
    RequestRecord,
    account,
    kill_via_command,
    run_fault_injection,
)


def _rec(rid: str, disposition: Disposition, attempts: int = 1, error=None,
         sent=0.0, ended=1.0, after_kill=False) -> RequestRecord:
    return RequestRecord(
        request_id=rid,
        intended_send_time=sent,
        disposition=disposition,
        error=error,
        attempts=[
            Attempt(index=i, target="http://x", sent_at=sent, ended_at=ended,
                    error=error if i == attempts - 1 else "transient",
                    after_kill=after_kill)
            for i in range(attempts)
        ],
    )


# ---------------------------------------------------------------------------
# The identity
# ---------------------------------------------------------------------------


def test_identity_holds_on_a_fully_accounted_run():
    records = [
        _rec("a", Disposition.COMPLETED),
        _rec("b", Disposition.COMPLETED),
        _rec("c", Disposition.FAILED, error="HTTP 503: dead"),
        _rec("d", Disposition.TIMED_OUT, error="ReadTimeout: timed out"),
        _rec("e", Disposition.SHED),
    ]
    acc = account(records)
    assert acc.issued == 5
    assert acc.accounted == 5
    assert (acc.completed, acc.failed, acc.timed_out, acc.shed) == (2, 1, 1, 1)
    assert acc.unaccounted == 0
    assert acc.to_dict()["identity_holds"] is True


def test_a_silently_dropped_request_is_an_error_not_a_rounding_difference():
    """
    THE test. A request with no terminal disposition is exactly what a failover
    bug produces, and it must be impossible to report a clean run over it.
    """
    records = [
        _rec("a", Disposition.COMPLETED),
        _rec("ghost", Disposition.PENDING),
        _rec("c", Disposition.COMPLETED),
    ]
    with pytest.raises(AccountingError) as exc:
        account(records)
    assert "ghost" in str(exc.value)
    assert "no terminal disposition" in str(exc.value)


def test_the_error_names_the_offending_ids():
    records = [_rec(f"g{i}", Disposition.PENDING) for i in range(3)]
    with pytest.raises(AccountingError) as exc:
        account(records)
    for i in range(3):
        assert f"g{i}" in str(exc.value)


def test_pending_is_representable_which_is_why_it_is_detectable():
    assert Disposition.PENDING.value == "pending"
    assert Disposition.PENDING not in {
        Disposition.COMPLETED, Disposition.FAILED,
        Disposition.TIMED_OUT, Disposition.SHED,
    }


# ---------------------------------------------------------------------------
# Retries are attempts, not outcomes
# ---------------------------------------------------------------------------


def test_retries_do_not_inflate_the_outcome_counts():
    records = [
        _rec("a", Disposition.COMPLETED, attempts=3),
        _rec("b", Disposition.COMPLETED, attempts=1),
    ]
    acc = account(records)
    assert acc.issued == 2
    assert acc.completed == 2, "a retried request is still ONE request"
    assert acc.attempts_total == 4
    assert acc.retried == 1
    assert acc.completed_after_retry == 1
    assert acc.accounted == acc.issued


def test_a_request_that_failed_every_attempt_counts_once():
    acc = account([_rec("a", Disposition.FAILED, attempts=3, error="HTTP 502: x")])
    assert acc.failed == 1 and acc.attempts_total == 3


# ---------------------------------------------------------------------------
# Kill-window bookkeeping
# ---------------------------------------------------------------------------


def test_in_flight_at_kill_counts_the_population_the_claim_is_about():
    records = [
        _rec("before", Disposition.COMPLETED, sent=0.0, ended=1.0),   # closed before
        _rec("spanning", Disposition.FAILED, sent=4.0, ended=6.0,     # open at kill
             error="ConnectError: refused"),
        _rec("after", Disposition.COMPLETED, sent=7.0, ended=8.0),    # opened after
    ]
    acc = account(records, kill_time=5.0)
    assert acc.in_flight_at_kill == 1
    assert acc.kill_time == 5.0


def test_recovery_time_measured_from_the_kill():
    records = [
        _rec("r", Disposition.COMPLETED, sent=6.0, ended=6.5, after_kill=True),
    ]
    acc = account(records, kill_time=5.0)
    assert acc.recovery_s == pytest.approx(1.5)


def test_no_post_kill_completion_reports_none_rather_than_zero():
    acc = account([_rec("a", Disposition.COMPLETED, sent=0.0, ended=1.0)], kill_time=5.0)
    assert acc.recovery_s is None


def test_errors_are_bucketed_with_bounded_cardinality():
    records = [
        _rec("a", Disposition.FAILED, error="ConnectError: connection refused"),
        _rec("b", Disposition.FAILED, error="HTTP 503: replica unhealthy"),
        _rec("c", Disposition.TIMED_OUT, error="ReadTimeout: timed out"),
    ]
    acc = account(records)
    assert set(acc.errors_by_kind) <= {
        "timeout", "connection_refused", "connection_reset",
        "connection_closed", "server_5xx", "client_4xx", "other",
    }
    assert sum(acc.errors_by_kind.values()) == 3


# ---------------------------------------------------------------------------
# End to end against a fake replica that dies
# ---------------------------------------------------------------------------


def _sse(chunks: list[dict]) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


class FakeReplica:
    """Serves SSE until `alive` goes false, then refuses connections."""

    def __init__(self):
        self.alive = True
        self.served = 0
        self.refused = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if not self.alive:
            self.refused += 1
            raise httpx.ConnectError("connection refused", request=request)
        self.served += 1
        return httpx.Response(200, content=_sse([
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]), headers={"content-type": "text/event-stream"})


def test_end_to_end_run_accounts_for_every_request_when_the_replica_dies():
    replica = FakeReplica()
    cfg = FaultConfig(
        url="http://replica/v1/chat/completions",
        rate=200.0, duration_s=0.2, kill_at_s=0.05, retries=1,
        retry_delay_s=0.0, request_timeout_s=2.0,
    )

    async def go():
        transport = httpx.MockTransport(replica.handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await run_fault_injection(
                cfg, killer=lambda: setattr(replica, "alive", False) or {"ok": True},
                client=client,
            )

    acc, records, kill_info = asyncio.run(go())

    assert kill_info["fired"] is True
    assert replica.refused > 0, "the kill must actually have taken the replica out"
    assert acc.issued == len(records) > 0
    # THE assertion: no request is silently dropped, whatever happened to it.
    assert acc.accounted == acc.issued
    assert acc.unaccounted == 0
    assert acc.completed > 0, "requests before the kill must have completed"
    assert acc.failed > 0, "requests after the kill must be counted as failures"
    for rec in records:
        assert rec.disposition is not Disposition.PENDING
        assert rec.attempts, f"{rec.request_id} has a disposition but no attempt"


def test_a_replica_that_reports_finish_reason_error_is_not_a_completion():
    """
    The fatal-CUDA path in serving/server/app.py ends every in-flight stream with
    an error chunk over a 200 response. Counting that as a completion because the
    HTTP status was fine is exactly the mistake that would hide R10 from the
    failover accounting.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse([
            {"choices": [{"delta": {"content": "plausible"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "error"}]},
        ]), headers={"content-type": "text/event-stream"})

    cfg = FaultConfig(url="http://r/v1/chat/completions", rate=50.0,
                      duration_s=0.05, kill_at_s=10.0, retries=0)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_fault_injection(cfg, killer=None, client=client)

    acc, records, _ = asyncio.run(go())
    assert acc.completed == 0
    assert acc.failed == acc.issued
    assert acc.accounted == acc.issued


def test_shed_429_is_not_retried_and_not_counted_as_a_fault():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"type": "queue_full"}})

    cfg = FaultConfig(url="http://r/v1/chat/completions", rate=50.0,
                      duration_s=0.05, kill_at_s=10.0, retries=2)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_fault_injection(cfg, killer=None, client=client)

    acc, records, _ = asyncio.run(go())
    assert acc.shed == acc.issued
    assert acc.failed == 0
    assert acc.attempts_total == acc.issued, (
        "load shedding is a deliberate answer; retrying it would convert an "
        "intentional overload response into a failover statistic"
    )


def test_killer_from_command_is_a_callable_and_does_not_run_at_build_time():
    killer = kill_via_command("true")
    assert callable(killer)
    result = killer()
    assert result["command"] == "true" and result["returncode"] == 0


def test_accounting_serialises_for_the_artifact():
    d = Accounting(issued=2, completed=2).to_dict()
    assert d["identity_holds"] is True
    json.dumps(d)  # must be artifact-writable
