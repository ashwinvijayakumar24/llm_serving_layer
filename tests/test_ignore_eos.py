"""
Output length must be CONTROLLED, not model-determined (methodology §4).

Job 11599377 asked for 64 tokens and averaged 12 (max 55) because the model
emitted EOS early. Consequences, in order of severity:

  1. The benchmark was mostly measuring PREFILL while reporting decode
     throughput.
  2. Worse: a scheduling change that alters which token gets sampled also alters
     how much work each request represents. The continuous-vs-static comparison
     would then differ in workload as well as in scheduling, and the delta would
     be attributable to neither.

Nothing errored. Requests completed, tokens streamed, throughput looked
plausible. These tests exist so the control cannot silently stop working.
"""

import pytest

from serving.memory.allocator import BlockAllocator
from serving.scheduler.scheduler import EOS_IDS, Request, RequestState, Scheduler, SchedulerConfig

EOS = next(iter(EOS_IDS))


def sched():
    return Scheduler(None, None, BlockAllocator(128, 16), SchedulerConfig())


def test_eos_stops_generation_by_default():
    """Serving default: EOS ends the request. This is correct for real traffic."""
    s = sched()
    r = Request(request_id="a", prompt_ids=[1, 2], max_tokens=64)
    r.prefill_pos = 2
    s._apply(r, 1, EOS)
    assert r.state == RequestState.FINISHED
    assert len(r.output_ids) == 1


def test_ignore_eos_runs_to_max_tokens():
    """Benchmark control: EOS is ignored, so every request is the same size."""
    s = sched()
    r = Request(request_id="a", prompt_ids=[1, 2], max_tokens=5, ignore_eos=True)
    r.prefill_pos = 2
    for _ in range(5):
        s._apply(r, 1, EOS)
    assert r.state == RequestState.FINISHED, "must still stop at max_tokens"
    assert len(r.output_ids) == 5, f"expected exactly 5 tokens, got {len(r.output_ids)}"


def test_ignore_eos_still_respects_max_tokens():
    """The escape hatch must not become an infinite generator."""
    s = sched()
    r = Request(request_id="a", prompt_ids=[1], max_tokens=3, ignore_eos=True)
    r.prefill_pos = 1
    for _ in range(10):
        if r.is_terminal():
            break
        s._apply(r, 1, 42)
    assert len(r.output_ids) == 3


def test_the_two_modes_actually_differ():
    """
    Guard against the control silently becoming a no-op — the same failure shape
    as a baseline that behaves like the system under test.
    """
    out = []
    for ignore in (False, True):
        s = sched()
        r = Request(request_id="a", prompt_ids=[1], max_tokens=8, ignore_eos=ignore)
        r.prefill_pos = 1
        for _ in range(8):
            if r.is_terminal():
                break
            s._apply(r, 1, EOS)
        out.append(len(r.output_ids))
    assert out == [1, 8], f"ignore_eos had no effect: got {out}"


def test_loadgen_defaults_to_controlled_output():
    """A benchmark must not opt IN to reproducibility; it must opt out."""
    from bench.loadgen import LoadGenConfig

    assert LoadGenConfig(url="http://x").ignore_eos is True


def test_server_schema_accepts_and_defaults_false():
    """The SERVING default is False — real traffic should honour EOS."""
    from serving.server.app import ChatCompletionRequest

    body = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
    assert body.ignore_eos is False
    assert ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}], ignore_eos=True
    ).ignore_eos is True


@pytest.mark.parametrize("ignore", [False, True])
def test_request_field_plumbed(ignore):
    assert Request(request_id="a", prompt_ids=[1], ignore_eos=ignore).ignore_eos is ignore
