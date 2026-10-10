"""CPU reproduction of FINDINGS_OURS F-004 / F-006 / F-007, without a GPU.

Drives the REAL scheduler, allocator and radix cache from serving/ (unmodified)
with the tiny float64 paged model the CPU preemption tests use
(tests/test_preemption.py). Mirrors W4 v2 in miniature: a small KV pool, a
closed loop of a few concurrent requests whose total demand FITS the pool, and
history from earlier requests left in the prefix cache.

Run from the repo root:  python3 docs/xengine/repro/f004_f007_cpu_repro.py

Output on 2026-10-10 (macOS, CPU, repo d603ca9+):
  cache off: 48 finished, 0 preemptions, 0 failed, 32/32 blocks free at end
  cache on:  48 finished, 111 preemptions, 9 FAILED with "sequence outgrew the
             KV pool ... holds 10 of 32, and it is the only running request",
             0/32 blocks free at end
Other sizes (64/4/40/16, 128/4/96/24, ...) give the same pattern: 0 everything
with the cache off; tens to ~140 preemptions and >=1 failure with it on.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from serving.cache.radix import RadixCache  # noqa: E402
from serving.scheduler.scheduler import Request, RequestState  # noqa: E402
from tests.test_preemption import TinyPagedModel, make_stack  # noqa: E402

POOL_BLOCKS = 32  # x 4-token blocks = 128 tokens
CONCURRENCY = 2
PROMPT_LEN = 40
MAX_TOKENS = 16  # 2 x (40 + 16) = 112 tokens of demand, inside the 128-token pool
N_REQUESTS = 48


def run(cache_on: bool, seed: int = 0) -> dict:
    model = TinyPagedModel()
    alloc, backend, sched = make_stack(model, POOL_BLOCKS)
    if cache_on:
        sched.prefix_cache = RadixCache(alloc, block_copy=getattr(backend, "copy_block", None))
    rng = random.Random(seed)
    prompts = [[rng.randrange(1, 64) for _ in range(PROMPT_LEN)] for _ in range(N_REQUESTS)]

    issued = 0
    done: list[Request] = []

    def submit():
        nonlocal issued
        sched.add_request(
            Request(
                request_id=f"r{issued}",
                prompt_ids=prompts[issued],
                max_tokens=MAX_TOKENS,
                ignore_eos=True,
            )
        )
        issued += 1

    for _ in range(CONCURRENCY):
        submit()
    steps = 0
    seen = 0
    while (sched.running or sched.waiting) and steps < 20000:
        sched.step()
        steps += 1
        while seen < len(sched.finished):  # closed loop: refill as requests finish
            done.append(sched.finished[seen])
            seen += 1
            if issued < N_REQUESTS:
                submit()

    failed = [r for r in done if r.state == RequestState.FAILED]
    empty = [r for r in done if not r.output_ids]
    return {
        "cache": "on" if cache_on else "off",
        "finished": len(done),
        "preemptions": sched.preemption.total,
        "starvation_fallbacks": sched.preemption.starvation_fallbacks,
        "failed": len(failed),
        "finished_with_no_output": len(empty),
        "free_blocks_at_end": alloc.num_free,
        "example_error": failed[0].error if failed else None,
    }


if __name__ == "__main__":
    for on in (False, True):
        print(run(on))
