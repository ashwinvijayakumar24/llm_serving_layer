"""
CUDA error checking at declared points, and the policy that follows from finding one.

WHY THIS FILE EXISTS (docs/RISK_REGISTER.md R10, the phase plan §9)
-----------------------------------------------------------------
`CLAIMS_AUDIT.md:299` on the engine: *"No CUDA error checking anywhere... A
launch failure silently produces garbage rather than raising."* The kernel
bindings call `cudaDeviceSynchronize()` at `kernels/bindings.cpp:39,52,65` and
throw the return code away. CUDA's error model is STICKY and DEFERRED: a kernel
that fails to launch — bad launch configuration, an illegal address, an
out-of-memory at launch time — does not raise at the call site. The error is
latched on the context and the next API call that reports errors returns it. If
nothing ever asks, nothing ever reports, and the tensor that kernel was supposed
to write keeps whatever was in that memory.

**This is a correctness issue, not an availability one, and the difference is
the whole point.** For a batch-1 microbenchmark an unchecked launch failure
shows up as a visibly wrong token and someone notices. In a serving system it
does not. The forward pass still returns a tensor of the right shape, argmax
still picks a token id, the detokenizer still produces words, the SSE stream
still flows, and throughput is unchanged because no work was actually done. The
replica emits PLAUSIBLE TEXT AT FULL THROUGHPUT with every metric green. There
is no gauge that goes red, because from the host's point of view nothing went
wrong. That failure mode is indistinguishable from correct operation by
observation, which is exactly the class of bug this project treats as
disqualifying (compare R3, R4: a plausible-output bug invalidates every
correctness claim at once).

THE POLICY: ANY CUDA ERROR IS FATAL TO THE REPLICA
--------------------------------------------------
Not "log and continue", not "retry the step", not "drop the request and carry
on". Once an error is latched, the CUDA context is poisoned: every subsequent
API call on that context returns the same sticky error, and there is no in-process
reset. `cudaDeviceReset()` is not usable here — PyTorch caches allocations,
streams, and module handles against the destroyed context, so a reset leaves a
process holding dangling device pointers, which is a worse state than the one it
was meant to fix. The only recovery is a new process.

So `CudaGuard` does three things and refuses to do a fourth:

  1. Records the error with the CONTEXT NAME of the declared check point, so the
     log names the operation rather than reporting a bare `RuntimeError: CUDA
     error` from somewhere inside a forward pass.
  2. Latches `poisoned = True` permanently. Every later `check()` re-raises the
     ORIGINAL error rather than pretending a fresh sync succeeded — because on a
     poisoned context a later sync may return the same sticky error, a different
     one, or (after enough churn) appear to succeed, and none of those mean the
     replica recovered.
  3. Leaves the decision to fail in-flight work to the caller, which is the
     server: `serving/server/app.py`'s `SchedulerLoop` catches the fatal error,
     marks itself unhealthy, fails every in-flight stream EXPLICITLY with an
     error chunk, and stops stepping. `/health` then returns 503 with
     `fatal.kind == "cuda"` so a router quarantines the replica instead of
     continuing to send it traffic.

  4. It does NOT attempt to continue. A guard that swallowed the error to keep
     the replica serving would be strictly worse than no guard at all: it would
     convert a detectable fault into the exact silent-garbage mode the check
     exists to prevent, while adding the appearance of safety.

Failing the in-flight requests explicitly matters as much as the health flag. A
poisoned replica that simply stops stepping leaves every open SSE stream hanging
until the client times out, and a client timeout is attributed to latency, not
to a fault. An explicit error chunk per stream is what makes the fault appear in
the client's accounting (`bench/fault_injection.py` counts exactly this).

COST, AND WHY IT IS NOT PURELY A COST (R2)
------------------------------------------
`check_cuda_error` calls `torch.cuda.synchronize()`, which blocks the host until
the device queue drains. Run once per scheduler step, its cost is bounded by the
step's own device time — the host would have to wait for that work before the
next step's results were usable anyway. Run per kernel it would serialise
launches and is not offered here; `every_n_steps` exists so the frequency is a
declared parameter rather than an accident.

The synchronisation is also load-bearing for measurement. `docs/RISK_REGISTER.md`
R2: `Scheduler.step()`'s host-clock timing is honest only because the step
currently ends in a device->host copy (`torch.argmax(...).tolist()`) that forces
a sync. The moment sampling moves on-device — which is what `forward_varlen`
returning a device tensor invites — `time.perf_counter()` around the step starts
measuring KERNEL-LAUNCH QUEUEING instead of execution: the number gets faster,
nothing errors, and `step_duration_host_ms` becomes fiction. A sync per step
pins the host clock back to execution. So the same call buys error detection AND
the validity of the step-duration histogram, and removing it to "save time"
would silently damage both.

NO-CUDA BEHAVIOUR
-----------------
Without CUDA (CI, laptops, the CPU test suite) every entry point is a no-op that
returns False, so the same code path runs in tests and on the GPU. `enabled` is
resolved once at construction and can be forced for testing via injected
`sync_fn` / `available_fn`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CudaFatalError",
    "CudaGuard",
    "check_cuda_error",
    "cuda_is_available",
    "default_guard",
    "FATAL_POLICY",
]

FATAL_POLICY = (
    "Any CUDA error is FATAL TO THE REPLICA. A poisoned CUDA context cannot be "
    "recovered in-process: the error is sticky, and cudaDeviceReset() is unusable "
    "under PyTorch's caching allocator. The replica marks itself unhealthy, fails "
    "every in-flight request explicitly, stops stepping, and reports 503 from "
    "/health so a router quarantines and restarts it (RISK_REGISTER.md R10)."
)


class CudaFatalError(RuntimeError):
    """
    A CUDA error observed at a declared check point. Always fatal to the replica.

    Carries `context` — the name of the check point — because the raw CUDA error
    text names neither the operation nor the request. "CUDA error: an illegal
    memory access was encountered" is true of every kernel in the process;
    "scheduler.step" is actionable.
    """

    def __init__(self, context: str, cause: BaseException | str):
        self.context = context
        self.cause_text = str(cause)
        super().__init__(
            f"CUDA error at {context!r}: {self.cause_text}\n{FATAL_POLICY}"
        )


def cuda_is_available() -> bool:
    """True only if torch imports AND reports a usable CUDA device."""
    try:
        import torch
    except Exception:  # noqa: BLE001 — absence of torch is a valid state here
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — a driver-level failure is "no CUDA"
        return False


def _default_sync() -> None:
    import torch

    torch.cuda.synchronize()


def check_cuda_error(
    context: str,
    *,
    sync_fn: Callable[[], None] | None = None,
    available_fn: Callable[[], bool] | None = None,
) -> bool:
    """
    Force a synchronisation and surface any sticky CUDA error, named by `context`.

    Returns True if a synchronisation actually happened, False if there is no
    CUDA to check (the no-op path — CI, CPU tests). Raises `CudaFatalError` if
    the device reports an error.

    `torch.cuda.synchronize()` is the check because it is the operation that
    both drains the queue and reports the latched error. Checking without
    synchronising would only see errors from work that happened to have already
    completed, which is precisely the asynchronous-launch case that makes
    unchecked errors invisible in the first place.
    """
    available = available_fn or cuda_is_available
    if not available():
        return False
    sync = sync_fn or _default_sync
    try:
        sync()
    except Exception as exc:  # noqa: BLE001 — every failure here is fatal
        raise CudaFatalError(context, exc) from exc
    return True


@dataclass
class CudaGuard:
    """
    A step-gated CUDA error check with a latched fatal state.

    `every_n_steps`:
        1  — check after every scheduler step (the default, and what makes the
             host-clock step timing meaningful; see the module docstring on R2).
        n  — check every n-th step. Detection is delayed by at most n steps, and
             those steps' output is already suspect, so this trades correctness
             latency for host time. Declared, never accidental.
        0  — disabled. Only appropriate where there is no CUDA at all.

    `sync_fn` / `available_fn` exist so the fatal-error path is testable on CPU.
    A guard whose failure path could only be exercised by corrupting a real GPU
    would never be exercised.
    """

    every_n_steps: int = 1
    sync_fn: Callable[[], None] | None = None
    available_fn: Callable[[], bool] | None = None

    checks_total: int = field(default=0, init=False)
    errors_total: int = field(default=0, init=False)
    skipped_total: int = field(default=0, init=False)
    sync_time_total_s: float = field(default=0.0, init=False)
    fatal: CudaFatalError | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.every_n_steps < 0:
            raise ValueError("every_n_steps must be >= 0 (0 disables checking)")
        self._available = (self.available_fn or cuda_is_available)()

    # -- state --------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether there is a device to check at all. Resolved once, at construction."""
        return self._available

    @property
    def enabled(self) -> bool:
        return self._available and self.every_n_steps > 0

    @property
    def poisoned(self) -> bool:
        """
        Latched. Once true, never false again in this process.

        There is no "the error cleared" state, because a poisoned context that
        later appears to sync cleanly has not recovered — it has merely stopped
        being asked the question it already answered.
        """
        return self.fatal is not None

    # -- checking -----------------------------------------------------------

    def check(self, context: str) -> bool:
        """
        Check now, regardless of step gating. Raises `CudaFatalError` on error.

        Re-raises the ORIGINAL error if already poisoned, so the first fault is
        the one reported no matter how many times this is called afterwards.
        """
        if self.fatal is not None:
            raise self.fatal
        if not self._available:
            self.skipped_total += 1
            return False
        t0 = time.perf_counter()
        try:
            ran = check_cuda_error(
                context, sync_fn=self.sync_fn, available_fn=lambda: True
            )
        except CudaFatalError as exc:
            self.errors_total += 1
            self.sync_time_total_s += time.perf_counter() - t0
            self.fatal = exc
            raise
        self.sync_time_total_s += time.perf_counter() - t0
        self.checks_total += 1
        return ran

    def maybe_check(self, context: str, step: int) -> bool:
        """
        The declared serving-path check point: call once per scheduler step.

        `step` is the scheduler's own step counter, so gating is deterministic
        and reproducible across runs rather than dependent on wall time.
        """
        if not self.enabled:
            self.skipped_total += 1
            return False
        if self.every_n_steps > 1 and step % self.every_n_steps != 0:
            self.skipped_total += 1
            return False
        return self.check(context)

    # -- reporting ----------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """
        What `/health` and `/metrics` publish about the guard.

        `sync_note` travels with the numbers on purpose: a reader who sees
        checking disabled must be able to see, in the same payload, that
        `step_duration_host_ms` lost its synchronisation guarantee at the same
        moment (R2).
        """
        return {
            "available": self._available,
            "enabled": self.enabled,
            "every_n_steps": self.every_n_steps,
            "checks_total": self.checks_total,
            "errors_total": self.errors_total,
            "skipped_total": self.skipped_total,
            "sync_time_total_s": self.sync_time_total_s,
            "poisoned": self.poisoned,
            "context": self.fatal.context if self.fatal else None,
            "error": self.fatal.cause_text if self.fatal else None,
            "policy": FATAL_POLICY,
            "sync_note": (
                "Each check is a torch.cuda.synchronize(). At every_n_steps=1 that "
                "is also what keeps host-clock step timing a measure of execution "
                "rather than kernel-launch queueing (R2). Disabling the check "
                "silently weakens step_duration_host_ms as well."
            ),
        }


_DEFAULT: CudaGuard | None = None


def default_guard() -> CudaGuard:
    """Process-wide guard, for call sites with nowhere to hang one."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = CudaGuard()
    return _DEFAULT
