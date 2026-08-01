"""
Workload generator tests. Pure CPU: no GPU, no tokenizer, no network, no model.

WHY THESE TESTS ARE NOT OPTIONAL
--------------------------------
Every cache and routing number this project publishes is a function of the
workload that produced it (docs/BENCHMARK_METHODOLOGY.md §7: "a hit rate without
the workload's sharing distribution attached is not a number"). A generator bug
therefore does not produce a test failure or a stack trace — it produces a
plausible benchmark result that is wrong, which is R14's entire point.

The tests are organised around the specific ways that happens:

  1. Reproducibility   — a seed that is ignored turns five "independent
                         repetitions" into one repetition reported five times.
  2. Length shape      — a distribution that collapses toward constant removes
                         the phenomena the benchmark claims to study.
  3. Zero sharing      — the CONTROL. If it shares anything, the measured cost
                         of the cache when it never helps is not measured.
  4. System sharing    — the requested prefixes must actually appear, and the
                         realized rate must track the requested one.
  5. Conversational    — turn n must genuinely contain turn n-1's history.
  6. Adversarial       — the divergence sweep must cover EVERY offset in a
                         block. This is the property the structure exists for.
  7. Sweeps            — the realized statistic must respond monotonically to
                         the requested parameter, or the sweep axis is fiction.
  8. R14 reporting     — a deliberately degenerate config must be caught by the
                         realized report, which is the only thing that can.

    python3 -m pytest tests/test_workloads.py -q
"""

import math
import statistics

import pytest

from bench.workloads.generator import (
    DEFAULT_BLOCK_SIZE,
    LengthSpec,
    Workload,
    WorkloadConfig,
    _substream,
    arrival_times,
    draw_lengths,
    generate,
    realized_distribution,
    render,
    render_to_text,
    sweep_sharing_rate,
)

ALL_STRUCTURES = ("zero", "system", "conversational", "adversarial")


def cfg(**kw) -> WorkloadConfig:
    """A small, fast default. Overridden per test."""
    base = dict(
        n_requests=120,
        structure="system",
        sharing_rate=0.5,
        block_size=DEFAULT_BLOCK_SIZE,
        seed=1234,
        prompt=LengthSpec(dist="lognormal", mean=256, sigma=0.8),
        output=LengthSpec(dist="lognormal", mean=64, sigma=0.8),
        n_shared_prefixes=4,
        shared_prefix_tokens=128,
        max_turns=6,
    )
    base.update(kw)
    return WorkloadConfig(**base)


def first_blocks(w: Workload) -> list[tuple[int, ...]]:
    bs = w.config.block_size
    return [r.token_ids[:bs] for r in w.requests if len(r.token_ids) >= bs]


def common_prefix_len(a, b) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


# ---------------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("structure", ALL_STRUCTURES)
def test_same_seed_gives_an_identical_workload(structure):
    """
    Byte-for-byte identical, not statistically similar. Token ids, max_tokens,
    and every piece of sharing provenance.
    """
    a = generate(cfg(structure=structure, seed=7))
    b = generate(cfg(structure=structure, seed=7))

    assert a.fingerprint() == b.fingerprint()
    assert len(a) == len(b) == 120
    for ra, rb in zip(a.requests, b.requests, strict=True):
        assert ra.token_ids == rb.token_ids
        assert ra.max_tokens == rb.max_tokens
        assert ra.shared_prefix_id == rb.shared_prefix_id
        assert ra.divergence_offset == rb.divergence_offset
        assert ra.parent_index == rb.parent_index
    assert a.realized == b.realized


@pytest.mark.parametrize("structure", ALL_STRUCTURES)
def test_different_seed_gives_a_different_workload(structure):
    a = generate(cfg(structure=structure, seed=7))
    b = generate(cfg(structure=structure, seed=8))
    assert a.fingerprint() != b.fingerprint()
    assert [r.token_ids for r in a.requests] != [r.token_ids for r in b.requests]


def test_length_and_structure_streams_are_independent():
    """
    Changing the sharing rate must NOT perturb the length draws, or a sharing
    sweep is secretly also a length sweep and the two effects are inseparable in
    the results. This is why the generator uses named RNG substreams.
    """
    a = generate(cfg(sharing_rate=0.0))
    b = generate(cfg(sharing_rate=1.0))
    assert [r.requested_prompt_len for r in a.requests] == [
        r.requested_prompt_len for r in b.requests
    ]
    assert [r.max_tokens for r in a.requests] == [r.max_tokens for r in b.requests]


def test_substreams_with_different_labels_diverge():
    x = [_substream(3, "a").random() for _ in range(4)]
    y = [_substream(3, "b").random() for _ in range(4)]
    assert x != y
    assert x == [_substream(3, "a").random() for _ in range(4)]


def test_arrival_process_is_seeded_and_has_the_requested_rate():
    a = arrival_times(2000, rate_rps=50.0, process="poisson", seed=5)
    b = arrival_times(2000, rate_rps=50.0, process="poisson", seed=5)
    c = arrival_times(2000, rate_rps=50.0, process="poisson", seed=6)
    assert a == b
    assert a != c
    assert a == sorted(a)
    # Mean inter-arrival ~ 1/rate. 2000 samples -> ~2% standard error.
    mean_gap = a[-1] / (len(a) - 1)
    assert mean_gap == pytest.approx(1.0 / 50.0, rel=0.10)

    det = arrival_times(10, rate_rps=50.0, process="deterministic")
    assert det == [i / 50.0 for i in range(10)]
    # Poisson is burstier than deterministic at the SAME mean rate. That gap is
    # the point of having both (§4).
    gaps = [a[i + 1] - a[i] for i in range(len(a) - 1)]
    assert max(gaps) > 3.0 * (1.0 / 50.0)


# ---------------------------------------------------------------------------
# 2. Length distributions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dist", ["lognormal", "uniform", "fixed", "heavy_tail"])
def test_each_distribution_has_approximately_the_requested_mean(dist):
    """
    All four families preserve the requested mean, including heavy_tail, whose
    body mean is solved so the mixture mean comes out right. That is what makes a
    heavy_tail-vs-lognormal comparison at the same mean a comparison of SHAPE
    rather than of load.

    Tolerance is wider for heavy_tail on purpose: with 2% of samples carrying 40%
    of the mass, the sample mean genuinely is noisier, and pretending otherwise
    would make this test flaky rather than correct.
    """
    n = 20_000
    spec = LengthSpec(dist=dist, mean=512, sigma=0.8)
    values = draw_lengths(n, spec, _substream(99, "t")).values
    d = realized_distribution(values)
    tol = 0.10 if dist == "heavy_tail" else 0.05
    assert d["mean"] == pytest.approx(512, rel=tol)
    assert d["n"] == n


def test_fixed_is_a_constant_control_and_the_others_are_not():
    fixed = realized_distribution(
        draw_lengths(500, LengthSpec(dist="fixed", mean=128), _substream(1, "t")).values
    )
    assert fixed["cv"] == 0.0
    assert fixed["min"] == fixed["max"] == 128

    for dist in ("lognormal", "uniform", "heavy_tail"):
        d = realized_distribution(
            draw_lengths(500, LengthSpec(dist=dist, mean=128), _substream(1, "t")).values
        )
        assert d["cv"] > 0.1, f"{dist} produced near-constant lengths"


def test_heavy_tail_actually_produces_a_heavy_tail():
    """
    Asserted RELATIVE to the lognormal at the same requested mean and sigma, not
    against a magic constant: the claim is that heavy_tail is materially more
    tail-heavy than the default, and a hardcoded threshold would silently stop
    testing that if the defaults moved.

    Why this matters beyond distribution trivia: one long generation holding its
    blocks while short requests queue behind it is the scenario that motivates
    preemption. A benchmark without such requests makes preemption look
    unnecessary, so the workload must be shown to contain them.
    """
    n = 20_000
    ln = realized_distribution(
        draw_lengths(n, LengthSpec(dist="lognormal", mean=512, sigma=0.8),
                     _substream(4, "t")).values
    )
    ht = realized_distribution(
        draw_lengths(n, LengthSpec(dist="heavy_tail", mean=512, sigma=0.8),
                     _substream(4, "t")).values
    )
    assert ht["p99_over_p50"] > 3.0 * ln["p99_over_p50"]
    # And the tail is long in absolute terms relative to the typical request:
    # the longest generation must hold blocks for far longer than the median one.
    assert ht["max"] > 20 * ht["p50"]


def test_heavy_tail_mean_is_preserved_by_construction():
    """The mixture identity (1-f)*body + f*mult*mean == mean, checked directly."""
    spec = LengthSpec(dist="heavy_tail", mean=1000, tail_fraction=0.05,
                      tail_multiplier=10.0, sigma=0.6, tail_sigma=0.4)
    d = realized_distribution(draw_lengths(40_000, spec, _substream(11, "t")).values)
    assert d["mean"] == pytest.approx(1000, rel=0.10)


def test_clipping_is_counted_not_swallowed():
    """
    A heavy tail clipped at max_len is a heavy tail with its tail removed, and
    nothing raises. The clip fraction is the only evidence that happened.
    """
    spec = LengthSpec(dist="heavy_tail", mean=512, max_len=1024)
    draw = draw_lengths(4000, spec, _substream(12, "t"))
    assert draw.n_clipped_high > 0
    assert max(draw.values) == 1024
    fr = draw.clip_fractions()
    assert 0.0 < fr["clipped_high_fraction"] < 1.0

    lo = draw_lengths(2000, LengthSpec(dist="lognormal", mean=4, sigma=1.5, min_len=8),
                      _substream(13, "t"))
    assert lo.n_clipped_low > 0
    assert min(lo.values) == 8


def test_output_length_is_controlled_per_request():
    """
    §4: output length is set from the distribution, not decided by the model.
    Every request carries its own max_tokens, and the realized max_tokens
    distribution is the requested output distribution.
    """
    w = generate(cfg(n_requests=600, output=LengthSpec(dist="lognormal", mean=200, sigma=0.7)))
    assert all(r.max_tokens >= 1 for r in w.requests)
    assert len({r.max_tokens for r in w.requests}) > 50, "output lengths are not varying"
    assert w.realized["output_len"]["mean"] == pytest.approx(200, rel=0.20)
    assert w.config.suppress_eos is True
    assert "max_tokens set per request" in w.realized["output_length_control"]


def test_bad_distribution_parameters_are_rejected():
    with pytest.raises(ValueError, match="unknown length distribution"):
        draw_lengths(1, LengthSpec(dist="gaussian"), _substream(0, "t"))
    with pytest.raises(ValueError, match="sigma must be positive"):
        draw_lengths(1, LengthSpec(dist="lognormal", sigma=0.0), _substream(0, "t"))
    with pytest.raises(ValueError, match="must be < 1"):
        # A tail that carries more than the whole mean has no valid body.
        draw_lengths(1, LengthSpec(dist="heavy_tail", tail_fraction=0.2,
                                   tail_multiplier=10.0), _substream(0, "t"))
    with pytest.raises(ValueError, match="not a tail"):
        draw_lengths(1, LengthSpec(dist="heavy_tail", tail_multiplier=1.0),
                     _substream(0, "t"))


# ---------------------------------------------------------------------------
# 3. zero — the control
# ---------------------------------------------------------------------------


def test_zero_sharing_shares_no_block_aligned_prefix_block_at_all():
    """
    THE CONTROL, and the strongest statement the test suite makes: no two
    requests share even ONE block-aligned prefix block.

    Checked pairwise on the first block rather than through the summary
    statistic, so that a bug in the statistic cannot make the control pass. If
    the first blocks are all distinct, no deeper block-aligned prefix can match
    either, because a block-b prefix match implies a block-0 match.
    """
    w = generate(cfg(structure="zero", n_requests=300, sharing_rate=0.9))
    # Prompts shorter than one block have no first block at all; those cannot
    # share a block-aligned prefix by definition, and are covered by the
    # distinct-prompt check below.
    fb = first_blocks(w)
    assert len(fb) > 250
    assert len(set(fb)) == len(fb), "two requests shared a first block under zero sharing"
    assert w.realized["sharing"]["distinct_prompts"] == 300

    for i in range(len(w.requests)):
        for j in range(i + 1, min(i + 6, len(w.requests))):
            shared = common_prefix_len(w.requests[i].token_ids, w.requests[j].token_ids)
            assert shared < w.config.block_size

    s = w.realized["sharing"]
    assert s["realized_block_sharing_rate"] == 0.0
    assert s["realized_request_sharing_rate"] == 0.0
    assert s["shared_prompt_blocks"] == 0
    assert s["total_prompt_blocks"] > 0
    assert w.degeneracy_warnings == []


@pytest.mark.parametrize("rate", [0.0, 0.5, 1.0])
def test_zero_sharing_ignores_the_sharing_rate_by_definition(rate):
    """It is the control at every point of the sweep, not only at rate 0."""
    w = generate(cfg(structure="zero", n_requests=80, sharing_rate=rate))
    assert w.realized["sharing"]["realized_block_sharing_rate"] == 0.0
    assert "ignores sharing_rate" in w.realized["sharing_rate_note"]


# ---------------------------------------------------------------------------
# 4. system — the common production shape
# ---------------------------------------------------------------------------


def test_system_sharing_produces_the_requested_number_of_distinct_prefixes():
    w = generate(cfg(structure="system", n_requests=400, sharing_rate=1.0,
                     n_shared_prefixes=6))
    assert w.realized["distinct_shared_prefixes_used"] == 6
    assert {r.shared_prefix_id for r in w.requests} == set(range(6))
    # At rate 1.0 every request starts with one of the six prefixes, so exactly
    # six distinct first blocks exist in the whole workload.
    assert w.realized["sharing"]["distinct_first_blocks"] == 6
    # The shared region is 128 tokens = 8 blocks, so the deepest match is 8.
    assert w.realized["sharing"]["max_shared_prefix_blocks"] == 128 // DEFAULT_BLOCK_SIZE


@pytest.mark.parametrize("rate", [0.25, 0.5, 0.75])
def test_system_realized_sharing_rate_tracks_the_requested_one(rate):
    """
    The realized request-level rate sits slightly BELOW the requested one by
    construction: the first request to use each prefix has nobody earlier to
    share with. With n=800 and 4 prefixes that deficit is ~0.5%.
    """
    n, k = 800, 4
    w = generate(cfg(structure="system", n_requests=n, sharing_rate=rate,
                     n_shared_prefixes=k))
    realized = w.realized["sharing"]["realized_request_sharing_rate"]
    tol = 0.05 + k / n
    assert realized == pytest.approx(rate, abs=tol)
    assert realized <= rate + 1e-9
    assert w.degeneracy_warnings == []


def test_system_unshared_requests_really_are_unshared():
    w = generate(cfg(structure="system", n_requests=200, sharing_rate=0.5))
    shared = [r for r in w.requests if r.shares_prefix_by_construction]
    unshared = [r for r in w.requests if not r.shares_prefix_by_construction]
    assert shared and unshared
    bs = w.config.block_size
    unshared_first = [r.token_ids[:bs] for r in unshared]
    assert len(set(unshared_first)) == len(unshared_first)
    assert set(unshared_first).isdisjoint({r.token_ids[:bs] for r in shared})


def test_skewed_prefix_popularity_is_expressible():
    """
    §10 case 7 — a single very hot prefix creating a routing hotspot — is named
    as the most likely place for a genuinely bad result and therefore the most
    valuable one to measure. A generator that could only produce uniform
    popularity could not produce that result at all.
    """
    uni = generate(cfg(structure="system", n_requests=600, sharing_rate=1.0,
                       n_shared_prefixes=8, prefix_popularity="uniform"))
    zipf = generate(cfg(structure="system", n_requests=600, sharing_rate=1.0,
                        n_shared_prefixes=8, prefix_popularity="zipf", zipf_s=1.4))

    def top_share(w):
        counts = {}
        for r in w.requests:
            counts[r.shared_prefix_id] = counts.get(r.shared_prefix_id, 0) + 1
        return max(counts.values()) / len(w.requests)

    assert top_share(zipf) > 2.0 * top_share(uni)


def test_a_shared_prefix_shorter_than_a_block_is_reported_as_degenerate():
    """
    The quiet failure this catches: sharing_rate says 1.0, the prefix is genuinely
    shared, and a block-granularity cache still sees zero hits because the shared
    region never fills a block. Requested parameters look perfect.
    """
    w = generate(cfg(structure="system", n_requests=100, sharing_rate=1.0,
                     shared_prefix_tokens=8, block_size=16))
    assert w.realized["sharing"]["realized_block_sharing_rate"] == 0.0
    assert any("shorter than" in msg for msg in w.degeneracy_warnings)


# ---------------------------------------------------------------------------
# 5. conversational — the most favourable structure
# ---------------------------------------------------------------------------


def test_conversational_turn_n_contains_the_full_history_of_turn_n_minus_1():
    """
    Not "shares a long prefix with" — CONTAINS, exactly: the parent's prompt
    followed by the parent's assistant tokens, as a literal prefix of the child.
    """
    w = generate(cfg(structure="conversational", n_requests=300, sharing_rate=0.85,
                     max_turns=8))
    continuations = [r for r in w.requests if r.turn_index > 0]
    assert len(continuations) > 50, "sharing rate produced almost no multi-turn traffic"

    for child in continuations:
        parent = w.requests[child.parent_index]
        history = parent.token_ids + parent.assistant_tokens
        assert child.token_ids[: len(history)] == history
        assert child.history_len == len(history)
        assert len(child.token_ids) > len(history)      # the new user turn exists
        assert child.turn_index == parent.turn_index + 1
        assert child.conversation_id == parent.conversation_id


def test_conversational_history_accumulates_across_a_whole_conversation():
    """Turn n contains turn n-2's history too, transitively."""
    w = generate(cfg(structure="conversational", n_requests=400, sharing_rate=0.9,
                     max_turns=8))
    chains = {}
    for r in w.requests:
        chains.setdefault(r.conversation_id, []).append(r)
    deep = [c for c in chains.values() if len(c) >= 3]
    assert deep, "no conversation reached three turns"
    for chain in deep:
        chain.sort(key=lambda r: r.turn_index)
        for i in range(1, len(chain)):
            assert chain[i].token_ids[: len(chain[0].token_ids)] == chain[0].token_ids
            assert len(chain[i].token_ids) > len(chain[i - 1].token_ids)
        assert [r.turn_index for r in chain] == list(range(len(chain)))
        assert len(chain) <= w.config.max_turns


def test_conversational_is_labeled_as_the_most_favourable_structure():
    """
    §4 requires this structure to be labeled where its numbers appear. The label
    travels in the realized report so it reaches the artifact, not just the
    docstring — a hit rate copied out of an artifact must carry it.
    """
    w = generate(cfg(structure="conversational", n_requests=200, sharing_rate=0.9))
    note = w.realized["conversation"]["note"]
    assert "MOST FAVOURABLE" in note
    assert w.realized["sharing"]["realized_block_sharing_rate"] > 0.4
    assert "most favourable" in render(w).lower()


def test_conversational_at_rate_zero_degenerates_to_zero_sharing():
    """The sweep must be continuous at its lower end."""
    w = generate(cfg(structure="conversational", n_requests=150, sharing_rate=0.0))
    assert all(r.turn_index == 0 for r in w.requests)
    assert w.realized["sharing"]["realized_block_sharing_rate"] == 0.0
    assert w.realized["conversation"]["n_conversations"] == 150


# ---------------------------------------------------------------------------
# 6. adversarial — the reason this structure exists
# ---------------------------------------------------------------------------


def test_adversarial_divergence_lands_at_every_offset_within_a_block():
    """
    THE TEST THIS STRUCTURE EXISTS FOR.

    Offset 0 is a clean block-boundary divergence; offsets 1..block_size-1 leave a
    partially matched block, which is where partial-hit accounting and
    copy-on-write of a block held by more than one sequence actually happen. A
    sweep that covered only offsets 0 and 8 would still look like a sweep in the
    config and would exercise almost none of that.

    Checked twice, independently: the recorded offsets, and the ACTUAL token
    positions at which pairs of prompts in the same group diverge. The second
    check does not consult the recorded metadata at all, so a generator that
    labeled its offsets correctly while emitting the wrong tokens still fails.
    """
    bs = DEFAULT_BLOCK_SIZE
    common = 128
    w = generate(cfg(structure="adversarial", n_requests=200, sharing_rate=1.0,
                     n_shared_prefixes=1, block_size=bs,
                     adversarial_common_tokens=common,
                     prompt=LengthSpec(dist="lognormal", mean=512, sigma=0.5)))

    div = w.realized["divergence"]
    assert div["n_diverging_requests"] == 200
    assert div["offsets_seen"] == list(range(bs)), "offset sweep is not exhaustive"
    assert div["offsets_missing"] == []
    assert div["offset_coverage"] == 1.0
    assert div["block_boundary_divergences"] > 0
    assert div["partial_block_divergences"] > 0

    # Independent check from the tokens alone: for two prompts whose recorded
    # offsets are a < b, the true common prefix must be exactly common + a.
    by_offset = {}
    for r in w.requests:
        by_offset.setdefault(r.divergence_offset, r)
    assert set(by_offset) == set(range(bs))

    observed = set()
    for a in range(bs):
        for b in range(a + 1, bs):
            shared = common_prefix_len(by_offset[a].token_ids, by_offset[b].token_ids)
            assert shared == common + a, (
                f"offsets {a}/{b} diverged at {shared}, expected {common + a}"
            )
            observed.add(shared - common)
    assert observed == set(range(bs - 1))


def test_adversarial_offsets_are_recorded_at_the_right_absolute_position():
    bs = 8
    common = 64
    w = generate(cfg(structure="adversarial", n_requests=64, sharing_rate=1.0,
                     block_size=bs, adversarial_common_tokens=common,
                     n_shared_prefixes=2))
    for r in w.requests:
        if r.divergence_offset is None:
            continue
        assert 0 <= r.divergence_offset < bs
        assert r.divergence_position == common + r.divergence_offset
        assert r.divergence_position % bs == r.divergence_offset % bs
    assert w.realized["divergence"]["offsets_seen"] == list(range(bs))


def test_adversarial_partial_coverage_is_reported_not_hidden():
    """
    Fewer participating requests than block_size cannot cover the sweep. The
    report must say so rather than presenting an incomplete sweep as a sweep.
    """
    bs = 16
    w = generate(cfg(structure="adversarial", n_requests=5, sharing_rate=1.0,
                     block_size=bs))
    div = w.realized["divergence"]
    assert div["offset_coverage"] < 1.0
    assert len(div["offsets_missing"]) == bs - div["n_diverging_requests"]
    assert "coverage" in render(w)


def test_adversarial_non_participants_are_fully_unique():
    w = generate(cfg(structure="adversarial", n_requests=200, sharing_rate=0.4))
    outsiders = [r for r in w.requests if r.divergence_offset is None]
    assert outsiders
    bs = w.config.block_size
    assert len({r.token_ids[:bs] for r in outsiders}) == len(outsiders)


# ---------------------------------------------------------------------------
# 7. Sharing-rate sweeps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("structure", ["system", "conversational", "adversarial"])
def test_sharing_rate_sweep_moves_the_realized_statistic_monotonically(structure):
    """
    §4 requires results as a FUNCTION of sharing rate. That axis is only real if
    the realized statistic responds to it: a sweep whose realized sharing barely
    moves is a sweep that reports the same workload five times under five labels.
    """
    rates = [0.0, 0.25, 0.5, 0.75, 1.0]
    loads = sweep_sharing_rate(cfg(structure=structure, n_requests=400), rates)
    block_rates = [w.realized["sharing"]["realized_block_sharing_rate"] for w in loads]

    assert block_rates == sorted(block_rates), f"non-monotonic: {block_rates}"
    assert block_rates[0] == 0.0
    assert block_rates[-1] > block_rates[0]
    # And the ends must be far apart, not merely ordered.
    assert block_rates[-1] > 0.1
    assert all(w.config.sharing_rate == r for w, r in zip(loads, rates, strict=True))


def test_sweep_holds_the_length_distribution_constant():
    loads = sweep_sharing_rate(cfg(n_requests=200), [0.0, 0.5, 1.0])
    means = [w.realized["prompt_len"]["mean"] for w in loads]
    drawn = [[r.requested_prompt_len for r in w.requests] for w in loads]
    assert drawn[0] == drawn[1] == drawn[2]
    # Realized prompt length still rises slightly with sharing, because a shared
    # prefix imposes a floor. That is recorded, not hidden.
    assert means[-1] >= means[0]
    assert loads[-1].realized["prompt_len_floored_by_structure"] >= 0


# ---------------------------------------------------------------------------
# 8. Realized-vs-requested reporting (R14)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("structure", ALL_STRUCTURES)
def test_realized_report_is_complete_for_every_structure(structure):
    w = generate(cfg(structure=structure, n_requests=150, sharing_rate=0.6))
    r = w.realized
    for key in ("n_requests", "prompt_len", "output_len", "sharing",
                "prompt_clipping", "output_clipping", "degeneracy_warnings"):
        assert key in r
    for stat in ("mean", "stdev", "p50", "p90", "p99", "min", "max", "n", "cv"):
        assert stat in r["prompt_len"]
        assert stat in r["output_len"]
    for stat in ("realized_block_sharing_rate", "realized_request_sharing_rate",
                 "distinct_first_blocks", "max_shared_prefix_blocks",
                 "total_prompt_blocks"):
        assert stat in r["sharing"]

    # Both sides travel to the artifact. Storing only one loses the comparison,
    # and the comparison is the alarm.
    fields = w.artifact_fields()
    assert fields["requested"]["sharing_rate"] == 0.6
    assert fields["realized"] is r
    assert fields["fingerprint"] == w.fingerprint()


def test_realized_length_stats_are_accurate_against_the_tokens_themselves():
    """The report must describe the actual requests, not the draw that preceded them."""
    w = generate(cfg(structure="system", n_requests=200, sharing_rate=0.7))
    lens = [len(r.token_ids) for r in w.requests]
    d = w.realized["prompt_len"]
    assert d["n"] == len(lens)
    assert d["mean"] == pytest.approx(sum(lens) / len(lens))
    assert d["min"] == min(lens)
    assert d["max"] == max(lens)
    assert d["p50"] == pytest.approx(statistics.median(lens), abs=1.0)

    outs = [r.max_tokens for r in w.requests]
    assert w.realized["output_len"]["mean"] == pytest.approx(sum(outs) / len(outs))


def test_a_collapsed_length_distribution_is_caught_by_the_realized_stats():
    """
    THE R14 CASE. A lognormal with a sigma small enough to be constant in
    practice: the config says "lognormal, mean 256" and looks entirely healthy.
    Only the realized cv reveals that every request is the same size and the
    benchmark has removed the phenomenon it claims to study.
    """
    degenerate = cfg(
        structure="system",
        n_requests=300,
        prompt=LengthSpec(dist="lognormal", mean=256, sigma=1e-4),
        output=LengthSpec(dist="lognormal", mean=64, sigma=1e-4),
    )
    w = generate(degenerate)

    assert w.config.prompt.dist == "lognormal"          # the config looks fine
    assert w.realized["prompt_len"]["cv"] < 0.01        # the realization does not
    assert w.realized["output_len"]["cv"] < 0.01
    msgs = " ".join(w.degeneracy_warnings)
    assert "COLLAPSED TOWARD CONSTANT" in msgs
    assert "prompt length" in msgs and "output length" in msgs
    assert "!!" in render(w)


def test_a_clipped_heavy_tail_is_caught():
    """A heavy tail truncated by max_len stops being a heavy tail, silently."""
    w = generate(cfg(
        structure="zero",
        n_requests=400,
        prompt=LengthSpec(dist="heavy_tail", mean=256, max_len=280),
    ))
    msgs = " ".join(w.degeneracy_warnings)
    assert "clipped at max_len" in msgs
    assert "not a heavy tail" in msgs
    assert w.realized["prompt_clipping"]["clipped_high_fraction"] > 0.05


def test_an_all_identical_workload_is_caught():
    """
    Simulates the seeding bug directly: one shared prefix, rate 1.0, and a
    fixed length equal to the prefix length, so every prompt is the same tokens.
    """
    w = generate(cfg(
        structure="system",
        n_requests=50,
        sharing_rate=1.0,
        n_shared_prefixes=1,
        shared_prefix_tokens=64,
        prompt=LengthSpec(dist="fixed", mean=64),
        output=LengthSpec(dist="fixed", mean=32),
    ))
    # Prompts are prefix + a unique marker suffix, so they are NOT identical --
    # uniqueness is a construction here, and the report proves it.
    assert w.realized["sharing"]["distinct_prompts"] == 50
    assert w.realized["sharing"]["realized_block_sharing_rate"] > 0.5

    # The genuinely identical case is caught.
    from bench.workloads.generator import _degeneracy_warnings
    fake = dict(w.realized)
    fake["sharing"] = dict(w.realized["sharing"], distinct_prompts=1)
    assert any("IDENTICAL" in m for m in _degeneracy_warnings(w.config, fake))


def test_a_healthy_workload_reports_no_warnings():
    """Otherwise the warnings are noise and nobody will read them."""
    for structure in ALL_STRUCTURES:
        w = generate(cfg(structure=structure, n_requests=400, sharing_rate=0.6,
                         prompt=LengthSpec(dist="lognormal", mean=512, sigma=0.8)))
        assert w.degeneracy_warnings == [], f"{structure}: {w.degeneracy_warnings}"


def test_the_oracle_sharing_rate_bounds_any_real_cache():
    """
    realized_block_sharing_rate is the infinite-cache hit rate, so it must never
    exceed 1 and must equal shared/total exactly. It is published as the bound a
    measured hit rate is judged against.
    """
    for structure in ALL_STRUCTURES:
        s = generate(cfg(structure=structure, n_requests=200,
                         sharing_rate=1.0)).realized["sharing"]
        assert 0.0 <= s["realized_block_sharing_rate"] <= 1.0
        if s["total_prompt_blocks"]:
            assert s["realized_block_sharing_rate"] == pytest.approx(
                s["shared_prompt_blocks"] / s["total_prompt_blocks"]
            )
        assert "ORACLE" in s["definition"]


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("structure", ALL_STRUCTURES)
@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_single_request_workload(structure, rate):
    w = generate(cfg(structure=structure, n_requests=1, sharing_rate=rate))
    assert len(w) == 1
    r = w.requests[0]
    assert len(r.token_ids) > 0
    assert r.max_tokens >= 1
    # One request cannot share with anything earlier, whatever the rate says.
    assert w.realized["sharing"]["realized_block_sharing_rate"] == 0.0
    assert w.realized["sharing"]["realized_request_sharing_rate"] == 0.0
    assert render(w)


@pytest.mark.parametrize("structure", ALL_STRUCTURES)
def test_sharing_rate_zero_and_one_are_both_valid_endpoints(structure):
    lo = generate(cfg(structure=structure, n_requests=200, sharing_rate=0.0))
    hi = generate(cfg(structure=structure, n_requests=200, sharing_rate=1.0))
    assert lo.realized["sharing"]["realized_block_sharing_rate"] == 0.0
    if structure == "zero":
        assert hi.realized["sharing"]["realized_block_sharing_rate"] == 0.0
    else:
        assert hi.realized["sharing"]["realized_block_sharing_rate"] > 0.0


def test_empty_workload_does_not_explode():
    w = generate(cfg(n_requests=0))
    assert len(w) == 0
    assert w.realized["prompt_len"]["n"] == 0
    assert w.realized["sharing"]["realized_block_sharing_rate"] == 0.0
    assert render(w)


def test_prompts_shorter_than_a_block_report_zero_shareable_blocks():
    """
    Not a bug, but it must not read as "the cache is broken": with prompts under
    one block, block-granularity sharing cannot exist at all.
    """
    w = generate(cfg(structure="system", n_requests=50, sharing_rate=1.0,
                     block_size=64, shared_prefix_tokens=8,
                     prompt=LengthSpec(dist="fixed", mean=12)))
    assert w.realized["sharing"]["total_prompt_blocks"] == 0
    assert w.realized["sharing"]["realized_block_sharing_rate"] == 0.0
    assert any("cannot exist" in m or "shorter than" in m for m in w.degeneracy_warnings)


def test_block_size_one_is_legal():
    w = generate(cfg(structure="adversarial", n_requests=20, sharing_rate=1.0,
                     block_size=1, adversarial_common_tokens=32))
    assert w.realized["divergence"]["offsets_seen"] == [0]
    assert w.realized["divergence"]["offset_coverage"] == 1.0


def test_invalid_config_is_rejected_loudly():
    with pytest.raises(ValueError, match="unknown sharing structure"):
        generate(cfg(structure="deep"))
    with pytest.raises(ValueError, match="sharing_rate must be in"):
        generate(cfg(sharing_rate=1.5))
    with pytest.raises(ValueError, match="block_size must be"):
        generate(cfg(block_size=0))
    with pytest.raises(ValueError, match="n_shared_prefixes must be"):
        generate(cfg(n_shared_prefixes=0))


# ---------------------------------------------------------------------------
# 10. Tokens, not strings
# ---------------------------------------------------------------------------


def test_generation_needs_no_tokenizer_and_emits_token_ids():
    """
    The whole suite runs without transformers installed. Prefix sharing is an
    exact match over ids, so making it depend on a tokenizer would make every
    sharing claim a claim about that tokenizer instead.
    """
    w = generate(cfg(structure="system", n_requests=50))
    for r in w.requests:
        assert isinstance(r.token_ids, tuple)
        assert all(isinstance(t, int) for t in r.token_ids)
        assert all(0 <= t < w.config.vocab_size for t in r.token_ids)


def test_render_to_text_is_optional_and_uses_a_supplied_tokenizer():
    class FakeTokenizer:
        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    w = generate(cfg(n_requests=3))
    texts = render_to_text(w, tokenizer=FakeTokenizer())
    assert len(texts) == 3
    assert texts[0].split()[0] == str(w.requests[0].token_ids[0])

    with pytest.raises(ValueError, match="needs a tokenizer"):
        render_to_text(w)


def test_marker_space_exhaustion_raises_rather_than_colliding():
    """
    Uniqueness is a construction, not a probability. When the construction runs
    out of room it must say so — a silent wraparound would make two "unique"
    requests identical and quietly inflate every hit rate that follows.
    """
    tiny = cfg(structure="zero", n_requests=50, vocab_size=64, marker_space=2)
    with pytest.raises(ValueError, match="exhausted unique markers"):
        generate(tiny)


def test_fingerprint_detects_any_token_change():
    w = generate(cfg(n_requests=20))
    before = w.fingerprint()
    w.requests[3].token_ids = w.requests[3].token_ids[:-1] + (
        (w.requests[3].token_ids[-1] + 1) % w.config.vocab_size,
    )
    assert w.fingerprint() != before


def test_percentile_method_is_stated():
    """§5: tools disagree on interpolation and the disagreement shows in the tail."""
    d = realized_distribution([1, 2, 3, 4])
    assert d["percentile_method"] == "linear interpolation between order statistics"
    assert d["p50"] == pytest.approx(2.5)
    assert math.isfinite(d["p99_over_p50"])
