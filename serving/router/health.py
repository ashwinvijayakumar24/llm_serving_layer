"""
Replica health: detection, quarantine, retry classification, ramp-in, drain.

This file is docs/ARCHITECTURE.md §9.3 turned into code, step for step, and the
steps are numbered in the class below so the walkthrough and the implementation
cannot drift apart.

THE ORGANIZING PRINCIPLE APPLIES HERE TOO — THE ROUTER HOLDS HINTS, THE REPLICA
HOLDS TRUTH. `ReplicaPool.inflight` is what this router *believes* is
outstanding, not what any replica reports; `status` is a conclusion drawn from
probes and errors that are already a few hundred milliseconds old. Every one of
those beliefs can be wrong, and being wrong costs placement quality — with
exactly one exception, which is the reason this file is larger than a health
checker needs to be:

    **A wrong belief about liveness can cost a CLIENT'S RESPONSE, not just
    latency, and that is the one place where the hint-only argument does not
    save us.** So the request-failure path here does not try to be clever. It
    classifies what the client has already been told and refuses to retry
    anything that would contradict it (see `InFlightPhase`).

DETECTION USES BOTH SIGNALS, AND THEY CATCH DIFFERENT FAILURES
--------------------------------------------------------------
* **In-flight request errors and timeouts** (`on_request_failure`) are the fast
  signal, and the one that actually protects latency: by the time a periodic
  probe notices, every request routed in the interim has already been dispatched
  into a black hole. One failure is enough to quarantine by default, because a
  dispatched request failing at the transport is direct evidence from the
  request path itself.
* **Active probes** (`on_probe_result`) are the slow signal, and the one that
  catches the failure the fast signal cannot: a replica that is *hung but
  connected* accepts TCP, holds requests open, and errors nothing. N consecutive
  probe failures quarantine it. The replica's own `/health` returns 503 when its
  scheduler task is dead (`serving/server/app.py`), which is precisely this
  case.

Neither alone is sufficient, so both are implemented and both feed one state
machine.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from serving.router.policy import ReplicaView

__all__ = [
    "ReplicaTarget",
    "ReplicaStatus",
    "InFlightPhase",
    "InFlightRecord",
    "HealthConfig",
    "HealthEvent",
    "ReplicaRuntime",
    "ReplicaPool",
]


# ---------------------------------------------------------------------------
# Targets and states
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicaTarget:
    """Where a replica lives. One GPU, one process, one base URL."""

    replica_id: str
    url: str

    @property
    def chat_url(self) -> str:
        return f"{self.url.rstrip('/')}/v1/chat/completions"

    @property
    def health_url(self) -> str:
        return f"{self.url.rstrip('/')}/health"


class ReplicaStatus(StrEnum):
    HEALTHY = "healthy"
    """Eligible. May still be ramping in — see `ReplicaRuntime.weight`."""
    QUARANTINED = "quarantined"
    """Ineligible. Detected dead or hung; hints purged."""
    DRAINING = "draining"
    """Ineligible for NEW work; in-flight requests are allowed to finish."""
    DRAINED = "drained"
    """Draining finished, nothing in flight. Safe to terminate."""


class InFlightPhase(StrEnum):
    """
    How far a request got before its replica died. This is §9.3 step 3, and the
    three-way split is the whole point of the step.

    NOT_STARTED
        Selected and possibly queued at the replica, but the router never saw a
        response begin. The client has been told nothing. **Safe to retry.**

    STARTED_NO_TOKENS
        The replica accepted it and may have emitted a role chunk carrying no
        content, but no visible token has reached the client. Retrying produces
        a complete, correct response the client cannot distinguish from a
        first-try one. **Safe to retry.**

    PARTIALLY_STREAMED
        At least one non-empty `delta.content` has already been written to the
        client's socket. **CANNOT be transparently retried.** There is no KV
        replication, so a retry re-generates from the prompt: the client would
        receive those tokens twice, or — because the new replica has different
        cache state and the continuation is a different sample path — receive a
        different continuation stitched onto the tokens it already has. Both are
        silently wrong output, which is worse than a visible failure by exactly
        the margin that makes silent corruption the worst class of bug in this
        project. The stream is terminated with an explicit error event instead.

    Note that the boundary is *visible content*, not the first SSE chunk. The
    replica's first chunk carries `{"role": "assistant"}` and no content
    (`serving/server/app.py`), and the load generator's TTFT is defined on the
    first non-empty content chunk (methodology §2). Treating the role chunk as
    "streamed" would forfeit a retry that is entirely safe.
    """

    NOT_STARTED = "not_started"
    STARTED_NO_TOKENS = "started_no_tokens"
    PARTIALLY_STREAMED = "partially_streamed"

    @property
    def retryable(self) -> bool:
        return self is not InFlightPhase.PARTIALLY_STREAMED


@dataclass
class InFlightRecord:
    """One dispatched request, tracked for the sole purpose of §9.3 step 3."""

    request_id: str
    replica_id: str
    phase: InFlightPhase = InFlightPhase.NOT_STARTED
    content_chunks: int = 0
    attempts: int = 1
    started_at: float = 0.0

    def note_response_started(self) -> None:
        if self.phase is InFlightPhase.NOT_STARTED:
            self.phase = InFlightPhase.STARTED_NO_TOKENS

    def note_content(self, n: int = 1) -> None:
        """A chunk with non-empty `delta.content` went to the client. Point of no return."""
        if n <= 0:
            return
        self.content_chunks += n
        self.phase = InFlightPhase.PARTIALLY_STREAMED

    @property
    def retryable(self) -> bool:
        return self.phase.retryable


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HealthConfig:
    """Every threshold in §9.3, named and defaulted."""

    probe_interval_s: float = 1.0
    probe_timeout_s: float = 1.0
    probe_failures_to_quarantine: int = 3
    """N consecutive ACTIVE probe failures. Slow signal; catches hung-but-connected."""
    request_failures_to_quarantine: int = 1
    """Consecutive in-flight failures. Fast signal; 1 because it is direct evidence."""
    probe_successes_to_recover: int = 2
    """Consecutive successful probes before a quarantined replica is re-admitted."""

    # -- ramp-in (§9.3 step 5) ---------------------------------------------
    ramp_seconds: float = 30.0
    ramp_steps: int = 6
    ramp_initial_weight: float = 0.1
    ramp_on_start: bool = False
    """
    Whether replicas ramp at router startup too. Default False: at the start of
    a benchmark every replica is equally cold, so ramping them all buys nothing
    and would put a decaying transient inside the measurement window.
    """

    # -- retry (§9.3 step 4) -----------------------------------------------
    max_retries: int = 2
    retry_base_s: float = 0.05
    retry_max_s: float = 2.0
    retry_jitter: bool = True

    # -- backpressure -------------------------------------------------------
    max_inflight_per_replica: int = 64
    max_inflight_total: int | None = None


@dataclass(frozen=True)
class HealthEvent:
    """
    An append-only record of every state transition.

    Exists because S7 is "fault injection during a load run **with full request
    accounting**" (PHASE_PLAN §8). An availability story with no per-event trail
    is unauditable — "requests were re-routed" is a claim, and this list is what
    turns it into a number.
    """

    at: float
    replica_id: str
    kind: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"at": self.at, "replica_id": self.replica_id,
                "kind": self.kind, "detail": self.detail}


# ---------------------------------------------------------------------------
# Per-replica runtime state
# ---------------------------------------------------------------------------


@dataclass
class ReplicaRuntime:
    target: ReplicaTarget
    status: ReplicaStatus = ReplicaStatus.HEALTHY
    inflight: int = 0
    consecutive_probe_failures: int = 0
    consecutive_probe_successes: int = 0
    consecutive_request_failures: int = 0
    ramp_start: float | None = None
    last_probe_at: float | None = None
    last_error: str | None = None

    # accounting
    dispatched: int = 0
    completed: int = 0
    failed: int = 0
    quarantines: int = 0
    recoveries: int = 0

    @property
    def replica_id(self) -> str:
        return self.target.replica_id

    def weight(self, now: float, cfg: HealthConfig) -> float:
        """
        Ramp-in weight in (0, 1]. Monotonically non-decreasing while ramping.

        **§9.3 step 5 is the non-obvious step.** A restarted replica passes its
        health check within seconds and is, by every liveness definition,
        healthy — and its KV cache is empty. A cache-aware router that treats it
        as equal sends it traffic it serves *slowly*, and the resulting TTFT
        spike is attributed to the recovery rather than to the routing decision
        that caused it. "Healthy" and "warm" are different properties and only
        one of them is health-checked, so the second one is handled here by
        weight rather than by status.

        Quantized into `ramp_steps` rather than made continuous so that "it
        ramps" is an assertion a test can make about distinct, observable
        levels instead of about floating-point drift.
        """
        if self.status in (ReplicaStatus.QUARANTINED, ReplicaStatus.DRAINING,
                           ReplicaStatus.DRAINED):
            return 0.0
        if self.ramp_start is None:
            return 1.0
        frac = (now - self.ramp_start) / max(1e-9, cfg.ramp_seconds)
        if frac >= 1.0:
            return 1.0
        step = int(max(0.0, frac) * cfg.ramp_steps)
        w0 = cfg.ramp_initial_weight
        return min(1.0, w0 + (1.0 - w0) * (step / cfg.ramp_steps))

    def is_warm(self, now: float, cfg: HealthConfig) -> bool:
        return self.weight(now, cfg) >= 1.0


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


class ReplicaPool:
    """
    The router's model of the fleet: liveness, load, ramp state, and the retry
    policy that follows from them.

    Clock and RNG are INJECTED. Not for tidiness — a ramp-in test that had to
    sleep 30 real seconds would not be run, and a jitter test against an unseeded
    global RNG asserts nothing. Both are the difference between §9.3 being
    implemented and §9.3 being tested (R21: this is all pure logic, so PACE queue
    time never blocks it).

    `on_quarantine` is how policy state gets purged. The pool does not import a
    policy; the router wires `pool.on_quarantine = policy.purge_replica` so the
    dependency runs one way and `health.py` stays testable with no policy at all.
    """

    def __init__(
        self,
        targets: Sequence[ReplicaTarget],
        config: HealthConfig | None = None,
        *,
        time_fn: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        on_quarantine: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config or HealthConfig()
        self.time_fn = time_fn
        self.rng = rng if rng is not None else random.Random(0xC0FFEE)
        self.on_quarantine = on_quarantine
        self.events: list[HealthEvent] = []

        now = self.time_fn()
        self.replicas: dict[str, ReplicaRuntime] = {}
        for t in targets:
            rt = ReplicaRuntime(target=t)
            if self.config.ramp_on_start:
                rt.ramp_start = now
            self.replicas[t.replica_id] = rt

        self.total_dispatched = 0
        self.total_retries = 0
        self.total_shed = 0
        self.total_failed_midstream = 0

    # -- introspection ------------------------------------------------------

    def __contains__(self, replica_id: object) -> bool:
        return replica_id in self.replicas

    def get(self, replica_id: str) -> ReplicaRuntime:
        return self.replicas[replica_id]

    @property
    def inflight_total(self) -> int:
        return sum(r.inflight for r in self.replicas.values())

    def views(self, now: float | None = None) -> list[ReplicaView]:
        """
        The fleet as a routing policy sees it.

        Order is registration order and is stable, because "ties broken by
        rotation" is only a definition if the list a policy rotates over does not
        reshuffle between calls.
        """
        t = self.time_fn() if now is None else now
        cfg = self.config
        out: list[ReplicaView] = []
        for r in self.replicas.values():
            eligible = (
                r.status is ReplicaStatus.HEALTHY
                and r.inflight < cfg.max_inflight_per_replica
            )
            out.append(
                ReplicaView(
                    replica_id=r.replica_id,
                    inflight=r.inflight,
                    eligible=eligible,
                    weight=r.weight(t, cfg),
                    warm=r.is_warm(t, cfg),
                )
            )
        return out

    def eligible_ids(self) -> list[str]:
        return [v.replica_id for v in self.views() if v.eligible]

    def target(self, replica_id: str) -> ReplicaTarget:
        return self.replicas[replica_id].target

    # -- events -------------------------------------------------------------

    def _emit(self, replica_id: str, kind: str, detail: str = "") -> None:
        self.events.append(HealthEvent(self.time_fn(), replica_id, kind, detail))

    # -- in-flight accounting ----------------------------------------------

    def acquire(self, replica_id: str) -> None:
        r = self.replicas[replica_id]
        r.inflight += 1
        r.dispatched += 1
        self.total_dispatched += 1

    def release(self, replica_id: str) -> None:
        r = self.replicas.get(replica_id)
        if r is None:
            return
        r.inflight = max(0, r.inflight - 1)
        if r.status is ReplicaStatus.DRAINING and r.inflight == 0:
            r.status = ReplicaStatus.DRAINED
            self._emit(replica_id, "drained", "in-flight reached zero")

    # -- STEP 1a: active health checks --------------------------------------

    def on_probe_result(self, replica_id: str, ok: bool, detail: str = "") -> None:
        """
        One probe outcome. N consecutive failures quarantine (§9.3 step 1a).

        Consecutive, not cumulative: a single dropped probe on a busy node is
        noise, and a router that quarantined on it would take a healthy replica
        out of service under exactly the load where capacity matters most.
        """
        r = self.replicas.get(replica_id)
        if r is None:
            return
        r.last_probe_at = self.time_fn()
        cfg = self.config
        if ok:
            r.consecutive_probe_failures = 0
            r.consecutive_probe_successes += 1
            if (
                r.status is ReplicaStatus.QUARANTINED
                and r.consecutive_probe_successes >= cfg.probe_successes_to_recover
            ):
                self.recover(replica_id)
            return

        r.last_error = detail or "probe failed"
        r.consecutive_probe_successes = 0
        r.consecutive_probe_failures += 1
        if (
            r.status is ReplicaStatus.HEALTHY
            and r.consecutive_probe_failures >= cfg.probe_failures_to_quarantine
        ):
            self.quarantine(
                replica_id,
                f"{r.consecutive_probe_failures} consecutive probe failures: {r.last_error}",
            )

    # -- STEP 1b: in-flight errors ------------------------------------------

    def on_request_failure(self, replica_id: str, detail: str = "") -> bool:
        """
        A dispatched request errored or timed out. THE FAST SIGNAL (§9.3 step 1b).

        Returns whether this failure quarantined the replica. Faster than the
        probe path by up to a full probe interval, and that interval is measured
        in requests: every request routed to a dead replica between the death and
        the next probe is a request that pays the full client timeout.
        """
        r = self.replicas.get(replica_id)
        if r is None:
            return False
        r.failed += 1
        r.last_error = detail or "in-flight request failed"
        r.consecutive_request_failures += 1
        r.consecutive_probe_successes = 0
        if (
            r.status is ReplicaStatus.HEALTHY
            and r.consecutive_request_failures >= self.config.request_failures_to_quarantine
        ):
            self.quarantine(replica_id, f"in-flight failure: {r.last_error}")
            return True
        return False

    def on_request_success(self, replica_id: str) -> None:
        r = self.replicas.get(replica_id)
        if r is None:
            return
        r.completed += 1
        r.consecutive_request_failures = 0

    # -- STEP 2: quarantine + hint purge ------------------------------------

    def quarantine(self, replica_id: str, reason: str = "") -> None:
        """
        Stop routing here IMMEDIATELY and purge this replica's prefix hints.

        The purge is not cleanup. The replica's KV cache died with its process,
        so every hint naming it is now actively harmful: it steers requests
        toward a dead replica's prefixes, and after a restart it would steer the
        cache-expecting requests at the one replica guaranteed to be cold
        (§9.3 step 2, and `HintTable.purge_replica`).
        """
        r = self.replicas.get(replica_id)
        if r is None or r.status is ReplicaStatus.QUARANTINED:
            return
        r.status = ReplicaStatus.QUARANTINED
        r.consecutive_probe_successes = 0
        r.ramp_start = None
        r.quarantines += 1
        self._emit(replica_id, "quarantined", reason)
        if self.on_quarantine is not None:
            self.on_quarantine(replica_id)

    # -- STEP 5: recovery, ramped ------------------------------------------

    def recover(self, replica_id: str) -> None:
        """Re-admit a quarantined replica — COLD, and therefore ramped (§9.3 step 5)."""
        r = self.replicas.get(replica_id)
        if r is None:
            return
        r.status = ReplicaStatus.HEALTHY
        r.consecutive_probe_failures = 0
        r.consecutive_request_failures = 0
        r.ramp_start = self.time_fn()
        r.recoveries += 1
        self._emit(replica_id, "recovered", "ramping in from cold")

    # -- STEP 6: graceful drain --------------------------------------------

    def drain(self, replica_id: str, reason: str = "") -> None:
        """
        Stop accepting new work here; let in-flight requests finish (§9.3 step 6).

        Distinct from quarantine in both directions: a drained replica is not
        failing, so its in-flight requests are not cancelled and its hints are
        NOT purged — its cache is still warm and still correct, and if it is
        re-admitted (a deploy that rolls back) those hints are immediately
        useful again.
        """
        r = self.replicas.get(replica_id)
        if r is None or r.status in (ReplicaStatus.DRAINING, ReplicaStatus.DRAINED):
            return
        r.status = ReplicaStatus.DRAINED if r.inflight == 0 else ReplicaStatus.DRAINING
        self._emit(replica_id, r.status.value, reason or "drain requested")

    def drain_all(self, reason: str = "router shutdown") -> None:
        for rid in list(self.replicas):
            self.drain(rid, reason)

    @property
    def fully_drained(self) -> bool:
        return all(
            r.status is ReplicaStatus.DRAINED
            or (r.status is ReplicaStatus.DRAINING and r.inflight == 0)
            for r in self.replicas.values()
        )

    def undrain(self, replica_id: str) -> None:
        r = self.replicas.get(replica_id)
        if r is not None and r.status in (ReplicaStatus.DRAINING, ReplicaStatus.DRAINED):
            r.status = ReplicaStatus.HEALTHY
            self._emit(replica_id, "undrained", "re-admitted after drain")

    # -- STEP 4: jittered retry --------------------------------------------

    def retry_delay(self, attempt: int) -> float:
        """
        Full-jitter exponential backoff: `uniform(0, min(cap, base * 2**attempt))`.

        **JITTER IS THE POINT, NOT THE BACKOFF** (§9.3 step 4). When a replica
        dies holding twelve requests, the eligible ones all become retryable at
        the same instant and all compute the same "least loaded" answer. Without
        jitter they arrive together at that one replica and convert one replica's
        failure into a second replica's overload — a thundering herd the router
        inflicted on itself while handling a failure.

        Full jitter rather than "delay ± 10%" because the spread has to be
        comparable to the delay itself to decorrelate a burst; a narrow jitter
        band around a common mean is still a burst, just a slightly wider one.
        """
        cfg = self.config
        ceiling = min(cfg.retry_max_s, cfg.retry_base_s * (2 ** max(0, attempt)))
        if not cfg.retry_jitter:
            return ceiling
        return self.rng.uniform(0.0, ceiling)

    def classify_and_count(self, record: InFlightRecord) -> bool:
        """
        §9.3 step 3, as one call: may this request be retried transparently?

        Counts the un-retryable ones separately because "N requests were lost
        mid-stream" is the honest number a fault-injection run has to publish,
        and it is a different number from "N requests failed".
        """
        if record.retryable and record.attempts <= self.config.max_retries:
            self.total_retries += 1
            return True
        if not record.retryable:
            self.total_failed_midstream += 1
        return False

    # -- backpressure -------------------------------------------------------

    def has_capacity(self) -> bool:
        """
        Router-level admission. Load shedding is an ANSWER, not a failure.

        Queueing without bound under overload converts a throughput problem into
        an unbounded latency problem in which every request misses its SLO;
        shedding keeps the served fraction servable, and it is what holds goodput
        flat above the knee instead of letting it collapse (methodology §3).
        """
        cap = self.config.max_inflight_total
        if cap is not None and self.inflight_total >= cap:
            return False
        return any(v.eligible for v in self.views())

    def note_shed(self) -> None:
        self.total_shed += 1

    # -- reporting ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        now = self.time_fn()
        cfg = self.config
        return {
            "replicas": [
                {
                    "replica_id": r.replica_id,
                    "url": r.target.url,
                    "status": r.status.value,
                    "inflight": r.inflight,
                    "weight": r.weight(now, cfg),
                    "warm": r.is_warm(now, cfg),
                    "dispatched": r.dispatched,
                    "completed": r.completed,
                    "failed": r.failed,
                    "quarantines": r.quarantines,
                    "recoveries": r.recoveries,
                    "consecutive_probe_failures": r.consecutive_probe_failures,
                    "last_error": r.last_error,
                }
                for r in self.replicas.values()
            ],
            "inflight_total": self.inflight_total,
            "dispatched_total": self.total_dispatched,
            "retries_total": self.total_retries,
            "shed_total": self.total_shed,
            "failed_midstream_total": self.total_failed_midstream,
            "failed_midstream_note": (
                "Requests whose replica died AFTER visible tokens reached the client. "
                "NOT retried, by design (ARCHITECTURE §9.3 step 3): a transparent retry "
                "would duplicate or diverge from output the client already has."
            ),
            "events": [e.as_dict() for e in self.events[-200:]],
        }
