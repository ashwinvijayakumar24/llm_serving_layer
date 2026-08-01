"""
Baseline B2 (static batching) must actually behave differently.

A baseline that silently behaves like the system under test is the worst
benchmark error available: both configurations report the same number, the
comparison looks like a null result, and nothing errors. These tests exist so
that cannot happen.

CPU only — the scheduler's admission logic needs no model.
"""

import pytest

from serving.memory.allocator import BlockAllocator
from serving.scheduler.scheduler import Request, RequestState, Scheduler, SchedulerConfig


def make(static: bool, **kw):
    alloc = BlockAllocator(num_blocks=256, block_size=16)
    cfg = SchedulerConfig(static_batching=static, max_batch_size=4, **kw)
    return Scheduler(model=None, backend=None, allocator=alloc, config=cfg)


def add(sched, n, prompt_len=8):
    for i in range(n):
        sched.add_request(Request(request_id=f"r{i}", prompt_ids=list(range(prompt_len))))


def test_continuous_admits_into_a_running_batch():
    """The behaviour under test: a new request joins while others are running."""
    s = make(static=False)
    add(s, 2)
    assert s._admit() == 2
    add(s, 2)
    assert s._admit() == 2, "continuous batching must admit into a running batch"
    assert len(s.running) == 4


def test_static_refuses_to_admit_while_anything_runs():
    """
    B2: a batch runs to completion before the next is admitted.

    This is the ONLY difference between the two configurations. Kernels, paged
    memory manager, HTTP stack, tokenizer and model are identical, so any
    goodput delta is attributable to scheduling and nothing else.
    """
    s = make(static=True)
    add(s, 2)
    assert s._admit() == 2
    add(s, 2)
    assert s._admit() == 0, "static batching must NOT admit while a batch is running"
    assert len(s.running) == 2


def test_static_admits_the_next_wave_once_drained():
    """Once the running set empties, the next wave goes in — otherwise it deadlocks."""
    s = make(static=True)
    add(s, 2)
    s._admit()
    add(s, 2)
    for r in list(s.running):
        r.state = RequestState.FINISHED
    s._retire_finished()
    assert s._admit() == 2, "static batching deadlocked: never admitted the next wave"


def test_the_two_configurations_actually_differ():
    """
    The guard against a null-result benchmark. Same inputs, same sequence of
    calls; the admission counts MUST diverge.
    """
    seq = []
    for static in (False, True):
        s = make(static=static)
        add(s, 2)
        counts = [s._admit()]
        add(s, 2)
        counts.append(s._admit())
        seq.append(counts)
    assert seq[0] != seq[1], (
        f"static and continuous admitted identically ({seq[0]}); the baseline is a "
        "no-op and any comparison against it is meaningless"
    )


@pytest.mark.parametrize("static", [False, True])
def test_config_flag_is_plumbed(static):
    assert make(static=static).config.static_batching is static
