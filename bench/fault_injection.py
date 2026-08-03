"""
Fault-injection harness: kill a replica mid-run and account for EVERY request.

WHAT CLAIM THIS BACKS
---------------------
Phase 5's router claims "health checking, graceful draining, and failover"
(the phase plan §11, bullet 5). Failover is the one of the three that cannot be
demonstrated by inspection, and it is also the one where a plausible-looking demo
proves nothing. "I killed a replica and the load generator kept printing numbers"
is not evidence: a failover bug does not usually stop the run, it silently loses
some requests, and a throughput chart drawn over the survivors looks fine. The
lost ones are exactly the requests that were in flight on the replica that died —
the population the claim is about.

**So the deliverable of this file is an ACCOUNTING IDENTITY, not a latency
number:**

    issued == completed + failed + timed_out + shed

Every request issued has exactly one terminal disposition, and `account()`
RAISES if any request is left without one. A request that vanished — no response,
no error, no exception, dropped somewhere between the router and a dead replica —
is the failure this harness exists to catch, and the only way to catch it is to
enumerate the issued set and demand a disposition for each member, rather than
summarising the responses that happened to come back.

`retried` is tracked separately and deliberately does NOT appear in that identity.
A retry is an attempt, not an outcome; adding attempts to outcomes is how a
harness ends up reporting more completions than requests. Attempts are reported
alongside as `attempts_total`, so "the router retried 40 requests and 38 of them
completed elsewhere" is a statement this artifact can support.

WHY THIS IS A SEPARATE FILE FROM bench/loadgen.py
-------------------------------------------------
`loadgen.py` measures a HEALTHY system: its outcomes feed percentiles and goodput
over a declared steady-state window, and it has no retry policy because a retry
would corrupt the arrival schedule that its intended-dispatch timing depends on
(R1). A fault run is not a steady-state measurement at all — there is a
discontinuity in the middle of it by construction — so mixing the two would put a
window with a hole in it into the same code path that publishes percentiles.
Kept separate, and `loadgen.py` is not modified.

NO GPU, NO SERVER IMPORTS
-------------------------
This runs against a URL over HTTP. It can be pointed at a router, at a single
replica, or at a fake ASGI app in a test. The killer is an injected callable so
the same harness works with `kill -9` on a local pid, `scancel` on a Slurm step,
or an admin endpoint — and so the accounting logic is testable with no process to
kill at all.

USAGE
-----
    python3 bench/fault_injection.py --url http://127.0.0.1:8000/v1/chat/completions \\
        --rate 8 --duration 30 --kill-at 15 --kill-cmd "scancel --signal=KILL 12345.0" \\
        --retries 1 --write results/p5
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import random
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # runnable as `python3 bench/fault_injection.py`
    sys.path.insert(0, str(REPO_ROOT))

__all__ = [
    "Disposition",
    "Attempt",
    "RequestRecord",
    "Accounting",
    "FaultConfig",
    "AccountingError",
    "account",
    "kill_via_command",
    "run_fault_injection",
    "main",
]


class Disposition(StrEnum):
    """
    Terminal states. `PENDING` is deliberately in the enum and deliberately NOT
    terminal — it is the state a record is born in, so that a record which never
    reaches a terminal state is representable and therefore detectable. An
    accounting scheme with no way to spell "we do not know what happened to this
    one" reports 100% accounted-for by construction.
    """

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SHED = "shed"


TERMINAL = frozenset(
    {Disposition.COMPLETED, Disposition.FAILED, Disposition.TIMED_OUT, Disposition.SHED}
)


class AccountingError(AssertionError):
    """Raised when a request has no terminal disposition. The point of the harness."""


@dataclass
class Attempt:
    """One HTTP attempt. A retried request has more than one."""

    index: int
    target: str
    sent_at: float
    ended_at: float | None = None
    status_code: int | None = None
    tokens: int = 0
    finish_reason: str | None = None
    error: str | None = None
    after_kill: bool = False


@dataclass
class RequestRecord:
    """
    One logical request across all its attempts.

    `tokens_by_attempt` is kept rather than summed because a request that got 12
    tokens from a dying replica and then 64 from its replacement produced 76
    token events for one logical request, and quietly summing them would report a
    token count no single response ever had.
    """

    request_id: str
    intended_send_time: float
    disposition: Disposition = Disposition.PENDING
    attempts: list[Attempt] = field(default_factory=list)
    error: str | None = None

    @property
    def retried(self) -> bool:
        return len(self.attempts) > 1

    @property
    def spanned_kill(self) -> bool:
        """In flight when the replica died — the population the claim is about."""
        return any(a.after_kill for a in self.attempts) and len(self.attempts) > 1

    @property
    def tokens_by_attempt(self) -> list[int]:
        return [a.tokens for a in self.attempts]


@dataclass
class Accounting:
    """The result. Every field here is a count of REQUESTS except `attempts_total`."""

    issued: int = 0
    completed: int = 0
    failed: int = 0
    timed_out: int = 0
    shed: int = 0
    unaccounted: int = 0
    retried: int = 0
    attempts_total: int = 0
    completed_after_retry: int = 0
    in_flight_at_kill: int = 0
    kill_time: float | None = None
    first_error_after_kill_s: float | None = None
    recovery_s: float | None = None
    """
    Seconds from the kill to the first attempt that COMPLETED against a live
    target afterwards. None means nothing completed after the kill, which for a
    single-replica run is the expected result and for a router run is the
    failure.
    """
    errors_by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def accounted(self) -> int:
        return self.completed + self.failed + self.timed_out + self.shed

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["accounted"] = self.accounted
        d["identity_holds"] = self.accounted == self.issued and self.unaccounted == 0
        return d


def account(records: list[RequestRecord], kill_time: float | None = None) -> Accounting:
    """
    Reduce records to counts and ENFORCE the identity.

    Raises `AccountingError` if any request lacks a terminal disposition, naming
    the offending ids. It raises rather than reporting because an unaccounted
    request invalidates the run: the number of requests the system lost is the
    measurement, and a run that cannot say what happened to N of them has not
    measured it. A caveat in a footnote is not a substitute — that is the same
    reasoning `assert_comparable` uses in serving/metrics/artifact.py.
    """
    acc = Accounting(issued=len(records), kill_time=kill_time)
    missing: list[str] = []
    errors: Counter[str] = Counter()

    for rec in records:
        acc.attempts_total += len(rec.attempts)
        if rec.retried:
            acc.retried += 1
        if rec.disposition not in TERMINAL:
            missing.append(rec.request_id)
            acc.unaccounted += 1
            continue
        if rec.disposition is Disposition.COMPLETED:
            acc.completed += 1
            if rec.retried:
                acc.completed_after_retry += 1
        elif rec.disposition is Disposition.FAILED:
            acc.failed += 1
        elif rec.disposition is Disposition.TIMED_OUT:
            acc.timed_out += 1
        elif rec.disposition is Disposition.SHED:
            acc.shed += 1

        if rec.error:
            errors[_error_kind(rec.error)] += 1

    if kill_time is not None:
        for rec in records:
            for a in rec.attempts:
                if a.sent_at <= kill_time and (a.ended_at is None or a.ended_at >= kill_time):
                    acc.in_flight_at_kill += 1
                    break
        after = [
            a.ended_at
            for rec in records
            for a in rec.attempts
            if a.after_kill and a.error is not None and a.ended_at is not None
        ]
        if after:
            acc.first_error_after_kill_s = min(after) - kill_time
        done = [
            a.ended_at
            for rec in records
            for a in rec.attempts
            if a.after_kill and a.error is None and a.ended_at is not None
        ]
        if done:
            acc.recovery_s = min(done) - kill_time

    acc.errors_by_kind = dict(sorted(errors.items()))

    if missing:
        raise AccountingError(
            f"{len(missing)} of {len(records)} requests have no terminal disposition: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}\n"
            "A request that vanished is exactly the failure this harness exists to "
            "detect — a failover bug loses in-flight requests silently and the "
            "throughput chart drawn over the survivors looks fine. This run does not "
            "support a failover claim."
        )
    if acc.accounted != acc.issued:
        raise AccountingError(
            f"accounting identity violated: issued={acc.issued} but "
            f"completed+failed+timed_out+shed={acc.accounted}"
        )
    return acc


def _error_kind(error: str) -> str:
    """Coarse bucketing. Kept coarse on purpose — an error string per request is
    unbounded cardinality and the interesting distinction is only 'refused vs
    reset vs 5xx vs timeout'."""
    e = error.lower()
    for needle, kind in (
        ("timeout", "timeout"),
        ("timed out", "timeout"),
        ("connect", "connection_refused"),
        ("reset", "connection_reset"),
        ("closed", "connection_closed"),
        ("http 5", "server_5xx"),
        ("http 4", "client_4xx"),
    ):
        if needle in e:
            return kind
    return "other"


# ---------------------------------------------------------------------------
# The killer
# ---------------------------------------------------------------------------


def kill_via_command(command: str) -> Any:
    """
    Build a killer from a shell command (`kill -9 PID`, `scancel ...`, `docker kill`).

    A command rather than a built-in `os.kill` because the replica this harness
    is pointed at is normally in another process namespace — a Slurm step, a
    container, another node. Injecting the command keeps the harness honest about
    the fact that it does not own the replica's lifecycle.
    """

    def _kill() -> dict[str, Any]:
        proc = subprocess.run(shlex.split(command), capture_output=True, text=True)
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-500:],
            "stderr": proc.stderr[-500:],
        }

    return _kill


# ---------------------------------------------------------------------------
# Config and run
# ---------------------------------------------------------------------------


@dataclass
class FaultConfig:
    url: str
    rate: float = 4.0
    duration_s: float = 30.0
    kill_at_s: float = 15.0
    retries: int = 1
    retry_delay_s: float = 0.25
    retry_url: str | None = None
    """
    Where a retry goes. None means the same URL — correct when pointed at a
    router, which is the interesting configuration: the router is supposed to
    send the retry somewhere alive. Pointing this at a second replica directly
    tests the client's failover instead of the router's, and the artifact records
    which was tested.
    """
    max_tokens: int = 64
    prompt: str = "Describe a fault-tolerant serving system in a few sentences."
    model: str = "llama-3.2-1b-instruct"
    request_timeout_s: float = 60.0
    seed: int = 20260801
    ignore_eos: bool = True


async def _attempt(
    client: Any, cfg: FaultConfig, url: str, index: int, t0: float, killed: list[float | None]
) -> Attempt:
    """One streaming attempt. Errors are DATA, never exceptions — an exception
    escaping here would abort the dispatch loop and destroy the very accounting
    the run exists to produce."""
    at = Attempt(index=index, target=url, sent_at=time.perf_counter() - t0)
    at.after_kill = killed[0] is not None and at.sent_at >= killed[0]
    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": cfg.prompt}],
        "max_tokens": cfg.max_tokens,
        "stream": True,
        "temperature": 0.0,
        "ignore_eos": cfg.ignore_eos,
    }
    try:
        async with client.stream(
            "POST", url, json=payload, timeout=cfg.request_timeout_s
        ) as resp:
            at.status_code = resp.status_code
            if resp.status_code != 200:
                body = await resp.aread()
                at.error = f"HTTP {resp.status_code}: {body[:200]!r}"
                at.ended_at = time.perf_counter() - t0
                return at
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    at.error = f"unparseable SSE payload: {data[:120]!r}"
                    break
                choice = (obj.get("choices") or [{}])[0]
                if (choice.get("delta") or {}).get("content"):
                    at.tokens += 1
                if choice.get("finish_reason"):
                    at.finish_reason = choice["finish_reason"]
    except Exception as exc:  # noqa: BLE001 — an outcome, not an exception path
        at.error = f"{type(exc).__name__}: {exc}"
    at.ended_at = time.perf_counter() - t0
    if at.error is None and at.finish_reason == "error":
        # The replica told us it failed. `serving/server/app.py` emits this for
        # every in-flight stream when the scheduler loop dies — including the
        # fatal-CUDA-error path. Counting it as a completion because the HTTP
        # status was 200 and the stream ended cleanly is exactly the mistake this
        # harness must not make.
        at.error = "stream finished with finish_reason=error (replica reported failure)"
    return at


async def _one_request(
    client: Any, cfg: FaultConfig, rec: RequestRecord, t0: float, killed: list[float | None]
) -> None:
    """Drive one logical request through up to `retries` retries, then dispose it."""
    for i in range(cfg.retries + 1):
        url = cfg.url if i == 0 else (cfg.retry_url or cfg.url)
        at = await _attempt(client, cfg, url, i, t0, killed)
        rec.attempts.append(at)
        if at.error is None:
            rec.disposition = Disposition.COMPLETED
            return
        rec.error = at.error
        if at.status_code == 429:
            # Shedding is a deliberate answer, not a fault. Retrying it here
            # would let the harness convert an intentional overload response into
            # a failover statistic.
            rec.disposition = Disposition.SHED
            return
        if i < cfg.retries:
            await asyncio.sleep(cfg.retry_delay_s)
    rec.disposition = (
        Disposition.TIMED_OUT if _error_kind(rec.error or "") == "timeout"
        else Disposition.FAILED
    )


async def run_fault_injection(
    cfg: FaultConfig,
    killer: Any = None,
    client: Any = None,
) -> tuple[Accounting, list[RequestRecord], dict[str, Any]]:
    """
    Dispatch at `cfg.rate` for `cfg.duration_s`, fire `killer` at `cfg.kill_at_s`,
    account for everything.

    Dispatch is open-loop for the same reason as `bench/loadgen.py`: a closed
    loop would stop issuing requests the moment the replica died, so the window
    in which requests SHOULD have failed would contain no requests, and the
    harness would report a clean failover because it stopped looking.
    """
    import httpx

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient()

    rng = random.Random(cfg.seed)
    n = max(1, int(cfg.rate * cfg.duration_s))
    # Poisson arrivals, seeded — same process as loadgen, same reason: fixed
    # intervals would let the kill land in a predictable gap between requests.
    schedule: list[float] = []
    t = 0.0
    for _ in range(n):
        t += rng.expovariate(cfg.rate) if cfg.rate > 0 else 0.0
        if t > cfg.duration_s:
            break
        schedule.append(t)

    records = [
        RequestRecord(request_id=f"fi-{i:05d}", intended_send_time=s)
        for i, s in enumerate(schedule)
    ]
    killed: list[float | None] = [None]
    kill_info: dict[str, Any] = {"fired": False}
    t0 = time.perf_counter()

    async def _do_kill() -> None:
        await asyncio.sleep(max(0.0, cfg.kill_at_s))
        killed[0] = time.perf_counter() - t0
        kill_info["fired"] = True
        kill_info["at_s"] = killed[0]
        if killer is not None:
            try:
                kill_info["result"] = killer()
            except Exception as exc:  # noqa: BLE001
                kill_info["result"] = {"error": f"{type(exc).__name__}: {exc}"}

    kill_task = asyncio.create_task(_do_kill())
    tasks: list[asyncio.Task] = []
    try:
        for rec in records:
            delay = rec.intended_send_time - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(
                asyncio.create_task(_one_request(client, cfg, rec, t0, killed))
            )
        # Wait for every request task. `gather` rather than a timeout-and-move-on,
        # because abandoning a task would produce precisely the unaccounted
        # request this harness reports on — self-inflicted.
        await asyncio.gather(*tasks)
        if cfg.kill_at_s <= cfg.duration_s:
            await kill_task
        else:
            # Scheduled outside the run window, so it is not part of this run.
            # Recorded rather than silently skipped: a run in which the kill never
            # fired is not a failover measurement, and the artifact has to say so.
            kill_task.cancel()
            kill_info["skipped"] = (
                f"kill_at_s={cfg.kill_at_s} is beyond duration_s={cfg.duration_s}; "
                "no fault was injected and this run measures nothing about failover"
            )
    finally:
        kill_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await kill_task
        if owns_client:
            await client.aclose()

    acc = account(records, kill_time=killed[0])
    kill_info["killer"] = "none (dry run)" if killer is None else "injected"
    return acc, records, kill_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def render(acc: Accounting, cfg: FaultConfig, kill_info: dict[str, Any]) -> str:
    a = acc
    lines = [
        "fault injection — replica killed mid-run",
        f"  target            {cfg.url}",
        f"  retry target      {cfg.retry_url or cfg.url}"
        f"{'  (same URL: testing the ROUTER)' if not cfg.retry_url else ''}",
        f"  offered rate      {cfg.rate} req/s for {cfg.duration_s}s"
        f"   (a PARAMETER, not achieved throughput)",
        f"  kill at           {cfg.kill_at_s}s   fired={kill_info.get('fired')}",
        "",
        "  ACCOUNTING (every issued request has exactly one terminal disposition)",
        f"    issued          {a.issued}",
        f"    completed       {a.completed}",
        f"    failed          {a.failed}",
        f"    timed_out       {a.timed_out}",
        f"    shed (429)      {a.shed}",
        f"    ---------------  {a.accounted}  == issued: {a.accounted == a.issued}",
        f"    unaccounted     {a.unaccounted}   (any non-zero invalidates the run)",
        "",
        f"  attempts total    {a.attempts_total}",
        f"  retried requests  {a.retried}  (an attempt, not an outcome — not in the identity)",
        f"  completed after retry  {a.completed_after_retry}",
        f"  in flight at kill      {a.in_flight_at_kill}",
        f"  first error after kill {a.first_error_after_kill_s}",
        f"  recovery (s to first post-kill completion)  {a.recovery_s}",
        f"  errors by kind    {a.errors_by_kind}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[1])
    p.add_argument("--url", required=True)
    p.add_argument("--retry-url", default=None)
    p.add_argument("--rate", type=float, default=4.0)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--kill-at", type=float, default=15.0)
    p.add_argument("--kill-cmd", default=None,
                   help="shell command that kills the replica, e.g. 'kill -9 12345'")
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--retry-delay", type=float, default=0.25)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260801)
    p.add_argument("--write", default=None, help="directory for the JSON artifact")
    args = p.parse_args(argv)

    cfg = FaultConfig(
        url=args.url, retry_url=args.retry_url, rate=args.rate,
        duration_s=args.duration, kill_at_s=args.kill_at, retries=args.retries,
        retry_delay_s=args.retry_delay, max_tokens=args.max_tokens, seed=args.seed,
    )
    killer = kill_via_command(args.kill_cmd) if args.kill_cmd else None
    if killer is None:
        print("no --kill-cmd: DRY RUN, nothing will be killed. "
              "The accounting still runs and must still balance.", flush=True)

    try:
        acc, records, kill_info = asyncio.run(run_fault_injection(cfg, killer=killer))
    except AccountingError as exc:
        # Printed, not raised as a traceback: the message IS the result of the
        # run, and a traceback reads like a harness bug rather than a finding.
        print(f"RUN INVALID — accounting does not balance:\n{exc}", flush=True)
        return 1
    print(render(acc, cfg, kill_info))

    if args.write:
        out = Path(args.write)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"fault_injection_{int(time.time())}.json"
        path.write_text(json.dumps({
            "config": asdict(cfg),
            "kill": kill_info,
            "accounting": acc.to_dict(),
            "records": [
                {
                    "request_id": r.request_id,
                    "intended_send_time": r.intended_send_time,
                    "disposition": r.disposition.value,
                    "attempts": [asdict(a) for a in r.attempts],
                    "error": r.error,
                }
                for r in records
            ],
        }, indent=2))
        print(f"\nwrote {path}")
    return 0 if acc.unaccounted == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
