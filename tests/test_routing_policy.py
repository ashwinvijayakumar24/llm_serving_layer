"""
Routing policy tests. CPU ONLY, MOCK REPLICAS, no GPU, no network, no HTTP.

That is a design property, not a convenience. The routing policy is pure logic
(docs/RISK_REGISTER.md R21): multi-process GPU orchestration under Slurm is
fiddly and PACE queue time is real, so the policy is built and validated against
mock replicas — counters in a dict — and queue time never blocks router work.

A "replica" in this file is a `ReplicaView` plus an integer. Nothing here loads
weights, opens a socket, or touches CUDA.

WHAT THE §10 TESTS ARE
----------------------
The last section encodes docs/BENCHMARK_METHODOLOGY.md §10 — the workload
classes where prefix-aware routing is predicted to LOSE — as executable
assertions about the DIRECTION of the effect, written before any measurement on
real hardware. They are predictions, not aspirations. **If one of them turns out
to be wrong on real replicas, that is a finding to report, not a test to
delete**, and the same is true if one of them passes here for a reason that does
not survive contact with a real scheduler.

    python3 -m pytest tests/test_routing_policy.py -q
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from dataclasses import dataclass

import pytest

from serving.router.policy import (
    ConsistentHash,
    HintTable,
    LeastOutstanding,
    PrefixAware,
    PrefixAwareConfig,
    PrefixKeyer,
    ReplicaView,
    RoundRobin,
    RouteRequest,
    build_policy,
)

# ---------------------------------------------------------------------------
# Mock fleet
# ---------------------------------------------------------------------------


def views(inflight: list[int], *, eligible: list[bool] | None = None,
          weights: list[float] | None = None) -> list[ReplicaView]:
    """A fleet snapshot from raw counters. Order is stable — rotation depends on it."""
    n = len(inflight)
    elig = eligible or [True] * n
    w = weights or [1.0] * n
    return [
        ReplicaView(replica_id=f"r{i}", inflight=inflight[i], eligible=elig[i], weight=w[i],
                    warm=w[i] >= 1.0)
        for i in range(n)
    ]


def req(i: int, key: str | None = None) -> RouteRequest:
    return RouteRequest(request_id=f"q{i}", prefix_key=key)


@dataclass
class SimResult:
    """What a simulated run produced. Every §10 assertion is about these fields."""

    assignments: list[str]
    counts: Counter
    peak_inflight: dict[str, int]

    @property
    def max_share(self) -> float:
        total = sum(self.counts.values())
        return max(self.counts.values()) / total

    @property
    def count_spread(self) -> int:
        """
        Busiest minus idlest, counting replicas that received NOTHING as zero.

        Zero-filling is not pedantry: a policy that starves a replica entirely
        leaves it out of the Counter, and a spread computed over observed keys
        alone would report perfect balance for the most imbalanced run possible.
        """
        seen = [self.counts.get(r, 0) for r in self.peak_inflight]
        return max(seen) - min(seen)

    @property
    def peak(self) -> int:
        return max(self.peak_inflight.values())


def simulate(policy, keys, n_replicas: int = 4, service: int = 3) -> SimResult:
    """
    A discrete mock fleet: one arrival per tick, each occupying a replica for
    `service` ticks.

    `service` is the knob that puts the run above or below the knee. With
    `service < n_replicas` the fleet keeps up and in-flight counts stay near
    zero; with `service >> n_replicas` in-flight grows and load pressure is what
    the policy is actually trading against — which is the regime §10 case 3 is
    about.

    No clock, no sleep, no concurrency: `now` is the tick index, so every run is
    exactly reproducible and a TTL is expressed in ticks.
    """
    ids = [f"r{i}" for i in range(n_replicas)]
    inflight = dict.fromkeys(ids, 0)
    peak = dict.fromkeys(ids, 0)
    finishing: list[tuple[int, str]] = []
    assignments: list[str] = []

    for tick, key in enumerate(keys):
        pending: list[tuple[int, str]] = []
        for finish_tick, rid in finishing:
            if finish_tick <= tick:
                inflight[rid] -= 1
            else:
                pending.append((finish_tick, rid))
        finishing = pending

        fleet = [ReplicaView(replica_id=r, inflight=inflight[r]) for r in ids]
        r = req(tick, key)
        chosen = policy.select(r, fleet, float(tick))
        assert chosen in ids, chosen
        policy.on_assign(r, chosen, float(tick))
        inflight[chosen] += 1
        peak[chosen] = max(peak[chosen], inflight[chosen])
        finishing.append((tick + service, chosen))
        assignments.append(chosen)

    return SimResult(assignments, Counter(assignments), peak)


# ---------------------------------------------------------------------------
# B4 — round robin
# ---------------------------------------------------------------------------


def test_round_robin_is_strict_rotation():
    p = RoundRobin()
    picks = [p.select(req(i), views([0, 0, 0, 0]), float(i)) for i in range(9)]
    assert picks == ["r0", "r1", "r2", "r3", "r0", "r1", "r2", "r3", "r0"]


def test_round_robin_ignores_load_and_cache_because_it_is_a_weak_baseline():
    """
    B4 hands the next request to a replica with 100 outstanding requests if it is
    that replica's turn. Asserting this pins the baseline's weakness in place:
    methodology §6 says beating round-robin is table stakes, and a B4 that
    quietly became load-aware would inflate every "improvement over B4" number.
    """
    p = RoundRobin()
    fleet = views([100, 0, 0, 0])
    picks = [p.select(req(i, key="same"), fleet, float(i)) for i in range(4)]
    assert picks == ["r0", "r1", "r2", "r3"]


def test_round_robin_skips_ineligible_replicas():
    p = RoundRobin()
    fleet = views([0, 0, 0, 0], eligible=[True, False, False, True])
    picks = [p.select(req(i), fleet, float(i)) for i in range(4)]
    assert set(picks) == {"r0", "r3"}
    assert picks == ["r0", "r3", "r0", "r3"]


def test_no_eligible_replica_returns_none_rather_than_raising():
    fleet = views([0, 0], eligible=[False, False])
    for p in (RoundRobin(), LeastOutstanding(), PrefixAware(), ConsistentHash()):
        assert p.select(req(0, key="k"), fleet, 0.0) is None


# ---------------------------------------------------------------------------
# B5 — least outstanding, THE REAL BASELINE
# ---------------------------------------------------------------------------


def test_least_outstanding_picks_the_least_loaded():
    p = LeastOutstanding()
    assert p.select(req(0), views([5, 2, 9, 3]), 0.0) == "r1"


def test_least_outstanding_ties_are_broken_by_rotation():
    """
    Methodology §6/B5 is explicit: 'ties broken by rotation'. Without it, B5
    degenerates to 'always the first tied replica' — which under a fleet that is
    idle (every count zero, i.e. every request tied) is not a load balancer at
    all, and would make the B5 baseline artificially easy to beat.
    """
    p = LeastOutstanding()
    picks = [p.select(req(i), views([0, 0, 0, 0]), float(i)) for i in range(8)]
    assert picks == ["r0", "r1", "r2", "r3"] * 2

    # Partial ties: r1 and r3 tie at 2, r0/r2 are busier. Rotation cycles within
    # the tied set only.
    q = LeastOutstanding()
    fleet = views([7, 2, 7, 2])
    assert [q.select(req(i), fleet, float(i)) for i in range(4)] == ["r1", "r3", "r1", "r3"]


def test_least_outstanding_honours_ramp_weight():
    """A cold replica looks loaded: inflight / weight. §9.3 step 5 in one line."""
    p = LeastOutstanding()
    fleet = views([1, 0], weights=[1.0, 0.1])
    # r1 is idle but at weight 0.1 its effective load is 0/0.1 = 0 -> still picked.
    assert p.select(req(0), fleet, 0.0) == "r1"
    # One request in flight on a 0.1-weight replica is an effective load of 10.
    fleet = views([5, 1], weights=[1.0, 0.1])
    assert p.select(req(1), fleet, 0.0) == "r0"


def test_policies_are_deterministic_under_a_seed():
    rng = random.Random(20260801)
    keys = [f"k{rng.randrange(6)}" for _ in range(200)]
    for name in ("round_robin", "least_outstanding", "prefix_aware", "consistent_hash"):
        a = simulate(build_policy(name), keys)
        b = simulate(build_policy(name), keys)
        assert a.assignments == b.assignments, name


# ---------------------------------------------------------------------------
# Prefix keying
# ---------------------------------------------------------------------------


def test_prefix_key_is_block_aligned_and_bounded():
    k = PrefixKeyer(block_size=16, n_blocks=2)
    shared = list(range(100))

    # Same first 32 (= 2 blocks) tokens, different tails -> same key.
    assert k.key_from_tokens(shared[:40] + [999] * 5) == k.key_from_tokens(shared[:32] + [7] * 9)
    # Fewer than one block: nothing block-aligned to be affine about.
    assert k.key_from_tokens(list(range(15))) is None
    # A difference inside the keyed region changes the key.
    diverged = shared[:20] + [123] + shared[21:40]
    assert k.key_from_tokens(diverged) != k.key_from_tokens(shared[:40])


def test_prefix_key_from_messages_separates_roles():
    k = PrefixKeyer(block_size=4, n_blocks=4, chars_per_token=4.0)
    a = k.key_from_messages([{"role": "system", "content": "x" * 200}])
    b = k.key_from_messages([{"role": "user", "content": "x" * 200}])
    assert a is not None and a != b


def test_hint_table_ttl_purge_and_fanout():
    t = HintTable(ttl_s=10.0, fanout=2)
    t.put("k", "r0", now=0.0)
    assert t.get("k", now=5.0) == ["r0"]
    assert t.get("k", now=11.0) == []          # expired, and treated as absent

    t.put("k", "r0", now=0.0)
    t.put("k", "r1", now=1.0)
    assert set(t.get("k", now=2.0)) == {"r0", "r1"}   # fanout=2 replicates a hot key
    assert t.purge_replica("r0") == 1
    assert t.get("k", now=2.0) == ["r1"]
    assert t.get(None, now=2.0) == []


# ---------------------------------------------------------------------------
# Prefix-aware: affinity, and the load blend
# ---------------------------------------------------------------------------


def test_prefix_aware_sends_the_same_prefix_to_the_same_replica_when_load_permits():
    p = PrefixAware(PrefixAwareConfig(blend=0.7, saturation_inflight=8))
    fleet = views([0, 0, 0, 0])
    first = p.select(req(0, "sysprompt"), fleet, 0.0)
    p.on_assign(req(0, "sysprompt"), first, 0.0)

    # Same key, and the affine replica is now the BUSIEST — affinity still wins
    # while it is below saturation, which is exactly the trade the blend makes.
    for i in range(1, 5):
        fleet = views([3 if r == int(first[1:]) else 0 for r in range(4)])
        again = p.select(req(i, "sysprompt"), fleet, float(i))
        assert again == first
        p.on_assign(req(i, "sysprompt"), again, float(i))
    assert p.affinity_honoured == 4


def test_prefix_aware_does_NOT_use_affinity_when_the_affine_replica_is_saturated():
    """
    The other half of the blend, and the §10 case 3 mechanism in miniature: past
    `saturation_inflight` no prefill saving repays the queueing, so affinity is
    not offered and load decides.
    """
    p = PrefixAware(PrefixAwareConfig(blend=0.9, saturation_inflight=4))
    p.hints.put("hot", "r0", 0.0)

    assert p.select(req(0, "hot"), views([3, 0, 0, 0]), 0.0) == "r0"      # below the cap
    diverted = p.select(req(1, "hot"), views([4, 0, 0, 0]), 1.0)          # at the cap
    assert diverted != "r0"
    assert p.affinity_declined_saturated == 1


def test_prefix_aware_blend_is_a_real_knob_in_both_directions():
    """
    Same fleet, same hint, opposite decisions. This is the knob §10 case 3 says
    must exist: 'prefix-aware routing wins at low-to-moderate load and loses
    above the knee UNLESS the policy explicitly blends affinity with load'.
    """
    fleet = views([7, 0, 0, 0])
    affinity_heavy = PrefixAware(PrefixAwareConfig(blend=0.9, saturation_inflight=8))
    affinity_heavy.hints.put("k", "r0", 0.0)
    assert affinity_heavy.select(req(0, "k"), fleet, 0.0) == "r0"

    load_heavy = PrefixAware(PrefixAwareConfig(blend=0.2, saturation_inflight=8))
    load_heavy.hints.put("k", "r0", 0.0)
    assert load_heavy.select(req(0, "k"), fleet, 0.0) != "r0"


def test_prefix_aware_at_blend_zero_is_exactly_b5():
    """
    A blend parameter whose zero is not the baseline is a parameter nobody can
    interpret. At blend=0 the policy must reproduce `LeastOutstanding` decision
    for decision, so any measured delta is attributable to the blend alone.
    """
    rng = random.Random(7)
    keys = [f"k{rng.randrange(4)}" for _ in range(300)]
    pa = simulate(PrefixAware(PrefixAwareConfig(blend=0.0, saturation_inflight=None,
                                                load_scale=1e9)), keys)
    b5 = simulate(LeastOutstanding(), keys)
    assert pa.assignments == b5.assignments


def test_prefix_aware_falls_back_to_load_when_the_prompt_has_no_block_aligned_prefix():
    p = PrefixAware()
    assert p.select(req(0, None), views([4, 1, 9]), 0.0) == "r1"


# ---------------------------------------------------------------------------
# Stale hints: the organizing principle, asserted
# ---------------------------------------------------------------------------


def test_stale_hints_only_cost_placement_and_never_correctness():
    """
    THE ORGANIZING PRINCIPLE, as an assertion: the router holds hints, the
    replica holds truth.

    Every hint below is wrong in a different way — naming a quarantined replica,
    a draining one, a replica that no longer exists, or one that evicted the
    prefix long ago. Not one of them may produce an error or a selection outside
    the eligible set. The cost of being wrong is a cache miss on a live replica,
    and that is the entire reason this layer needs no consensus.
    """
    rng = random.Random(1234)
    p = PrefixAware(PrefixAwareConfig(blend=0.8, saturation_inflight=6))

    # Seed the table with hints for replicas that are about to be wrong, plus
    # some for replicas that were never in the fleet at all.
    for i in range(50):
        p.hints.put(f"k{i}", rng.choice(["r0", "r1", "r2", "r3", "ghost", "r99"]), 0.0)

    for i in range(500):
        elig = [rng.random() > 0.4 for _ in range(4)]
        if not any(elig):
            elig[rng.randrange(4)] = True
        fleet = views([rng.randrange(0, 12) for _ in range(4)], eligible=elig)
        eligible_ids = {v.replica_id for v in fleet if v.eligible}
        chosen = p.select(req(i, f"k{rng.randrange(60)}"), fleet, float(i))
        assert chosen in eligible_ids     # never a dead, draining, or unknown replica


def test_purging_a_replica_removes_every_hint_that_names_it():
    p = PrefixAware()
    for i in range(20):
        p.on_assign(req(i, f"k{i % 5}"), "r2", float(i))
    assert len(p.hints) == 5
    p.purge_replica("r2")
    assert all(p.hints.get(f"k{i}", 20.0) == [] for i in range(5))


def test_expired_hints_are_ignored_not_obeyed():
    p = PrefixAware(PrefixAwareConfig(blend=1.0, hint_ttl_s=10.0, saturation_inflight=None))
    p.on_assign(req(0, "k"), "r3", 0.0)
    assert p.select(req(1, "k"), views([0, 0, 0, 0]), 5.0) == "r3"
    # Past the TTL the hint is gone, so the choice reverts to load/rotation.
    # High cache turnover (§10 case 5) is exactly why a hint must decay on its own.
    stale_pick = p.select(req(2, "k"), views([0, 0, 0, 9]), 100.0)
    assert stale_pick != "r3"


# ---------------------------------------------------------------------------
# Consistent hash — the affinity-without-adaptation control
# ---------------------------------------------------------------------------


def test_consistent_hash_is_stable_per_key_and_spreads_distinct_keys():
    p = ConsistentHash(virtual_nodes=64)
    fleet = views([0] * 4)
    for _ in range(5):
        assert p.select(req(0, "key-a"), fleet, 0.0) == p.select(req(1, "key-a"), fleet, 0.0)
    picks = {p.select(req(i, f"key-{i}"), fleet, 0.0) for i in range(200)}
    assert len(picks) == 4


def test_consistent_hash_moves_only_the_lost_replicas_keys():
    """
    A quarantine that reshuffled every key would turn one replica's death into a
    fleet-wide cold start. Consistent hashing is here precisely so that
    comparison can be made.
    """
    p = ConsistentHash(virtual_nodes=128)
    full = views([0] * 4)
    before = {f"k{i}": p.select(req(i, f"k{i}"), full, 0.0) for i in range(400)}
    degraded = views([0] * 4, eligible=[True, True, True, False])
    after = {f"k{i}": p.select(req(i, f"k{i}"), degraded, 1.0) for i in range(400)}
    moved = [k for k in before if before[k] != after[k]]
    assert all(before[k] == "r3" for k in moved)


# ===========================================================================
# METHODOLOGY §10 — WHERE PREFIX-AWARE ROUTING SHOULD LOSE
#
# Predictions made BEFORE measurement, asserted as directions. Mandatory, not
# optional: a results table containing only wins is evidence of workload
# selection, and a reader who knows this field will assume exactly that.
# ===========================================================================


def test_s10_case1_zero_sharing_ties_b5():
    """
    §10 case 1 — every request has a unique prefix.

    There is nothing to be cache-aware about: no hint ever hits, so the affinity
    term is identically zero and the policy is B5 with extra bookkeeping.
    PREDICTION: ties B5. Asserted in the strongest available form — decision for
    decision, not merely on aggregate balance — because any deviation from
    load-optimal placement here is pure loss.

    (The one way this could fail on real replicas: above `load_scale` the load
    term saturates and stops distinguishing loads. That is a real, documented
    limit of the scoring function rather than an artefact of this test, and it
    is the reason `load_scale` is tunable.)
    """
    keys = [f"unique-{i}" for i in range(400)]
    pa = simulate(PrefixAware(PrefixAwareConfig(blend=0.7, saturation_inflight=None,
                                                load_scale=1e9)), keys)
    b5 = simulate(LeastOutstanding(), keys)
    assert pa.assignments == b5.assignments
    assert pa.count_spread == b5.count_spread


def test_s10_case2_uniform_sharing_degenerates_to_pick_any_and_b5_wins():
    """
    §10 case 2 — EVERY request shares the SAME prefix.

    After warmup every replica has cached it, so cache state is identical
    everywhere and prefix awareness degenerates to 'pick any replica'. The
    router, however, has one hint and follows it, piling the whole workload onto
    one replica. PREDICTION: B5 wins on load balance alone.

    **This case matters because it is the naive mental model of prefix
    caching** — 'requests share a system prompt, so route by prefix' — and it is
    the case where the fancy router is worthless or worse.
    """
    keys = ["one-shared-system-prompt"] * 240
    pa = simulate(PrefixAware(PrefixAwareConfig(blend=0.9, saturation_inflight=None)), keys)
    b5 = simulate(LeastOutstanding(), keys)

    assert pa.max_share > 0.95          # essentially everything on one replica
    assert b5.max_share < 0.30          # four replicas, evenly split
    assert pa.count_spread > b5.count_spread
    assert pa.peak > b5.peak            # and it shows up as queueing, not just counts


def test_s10_case3_above_the_knee_affinity_and_balance_conflict():
    """
    §10 case 3 — high load, near or above capacity.

    Affinity and load balance conflict DIRECTLY: the replica holding the popular
    prefix is busy *because* it holds the popular prefix. Routing for affinity
    concentrates load; routing for balance destroys affinity. PREDICTION:
    prefix-aware wins below the knee and loses above it unless the policy
    explicitly blends — so pure affinity must show materially worse peak
    queueing than B5 under a service time the fleet cannot keep up with.

    The third arm is the mitigation: the saturation cutoff must claw most of
    that back. Publishing where that crossover sits is a better result than
    publishing a win.
    """
    keys = ["hot-prefix"] * 120
    pure_affinity = simulate(
        PrefixAware(PrefixAwareConfig(blend=1.0, saturation_inflight=None)),
        keys, n_replicas=4, service=20,
    )
    b5 = simulate(LeastOutstanding(), keys, n_replicas=4, service=20)
    blended = simulate(
        PrefixAware(PrefixAwareConfig(blend=1.0, saturation_inflight=6)),
        keys, n_replicas=4, service=20,
    )

    assert pure_affinity.peak > b5.peak * 2       # the predicted loss, above the knee
    assert blended.peak < pure_affinity.peak      # blending recovers most of it
    assert blended.peak >= b5.peak                # but not all of it — affinity costs balance


def test_s10_case7_hot_prefix_skew_is_a_self_inflicted_hotspot():
    """
    §10 case 7 — highly skewed prefix popularity.

    One very hot prefix means affinity routing sends a disproportionate share of
    traffic to whichever replica owns it. PREDICTION (§10 calls this the most
    likely place for a genuinely bad result, and therefore the most valuable one
    to measure): the hot replica's share far exceeds 1/N, while B5's stays at
    1/N because it cannot see the prefix at all.
    """
    rng = random.Random(99)
    keys = ["HOT" if rng.random() < 0.6 else f"cold-{rng.randrange(40)}" for _ in range(400)]

    skewed = simulate(PrefixAware(PrefixAwareConfig(blend=0.95, saturation_inflight=None)),
                      keys, service=8)
    b5 = simulate(LeastOutstanding(), keys, service=8)

    assert skewed.max_share > 0.55           # >> 1/4: the hotspot the router inflicted
    assert skewed.max_share > 2 * b5.max_share
    assert b5.max_share < 0.30               # cache-blind, and therefore balanced
    assert skewed.peak > b5.peak * 2         # and the hotspot is queueing, not just counts

    # The mitigation, and its cost: the saturation cutoff spills the hot prefix
    # onto other replicas — buying balance by giving up the very affinity the
    # policy exists for.
    capped = simulate(PrefixAware(PrefixAwareConfig(blend=0.95, saturation_inflight=3)),
                      keys, service=8)
    assert capped.max_share < skewed.max_share
    assert capped.peak < skewed.peak


def test_s10_case4_short_prompts_have_no_prefix_to_be_aware_of():
    """
    §10 case 4 — short prompts.

    Not a scoring effect but a keying one, and it is the honest floor: a prompt
    shorter than one block produces NO key at all, so the policy is B5 by
    construction. There is a prompt length below which prefix awareness is a
    no-op, and it is exactly `block_size` tokens.
    """
    k = PrefixKeyer(block_size=16, n_blocks=4)
    assert k.key_from_tokens(list(range(15))) is None
    p = PrefixAware(PrefixAwareConfig(blend=1.0))
    picks = [p.select(req(i, None), views([0, 0, 0, 0]), float(i)) for i in range(8)]
    assert picks == ["r0", "r1", "r2", "r3"] * 2


def test_s10_case6_benefit_requires_routing_freedom():
    """
    §10 case 6 — very few replicas.

    At N=1 the policy is a no-op: there is nothing to route to and only the
    replica's own radix cache matters. Benefit should be expected to GROW with
    N, and any published result must state N for that reason.
    """
    p = PrefixAware(PrefixAwareConfig(blend=1.0))
    single = [p.select(req(i, f"k{i}"), views([i]), float(i)) for i in range(5)]
    assert set(single) == {"r0"}


# ---------------------------------------------------------------------------
# Interface hygiene
# ---------------------------------------------------------------------------


def test_build_policy_rejects_unknown_names_loudly():
    """
    A benchmark that ran B5 while its artifact said `prefix_aware` is the
    worst-shaped error this project can make: it looks like a null result rather
    than like a bug. Same reasoning as the SERVING_STATIC_BATCHING banner.
    """
    with pytest.raises(KeyError):
        build_policy("prefix_awear")


def test_every_policy_reports_stats_and_survives_a_purge():
    for p in (RoundRobin(), LeastOutstanding(), PrefixAware(), ConsistentHash()):
        p.purge_replica("r0")
        assert p.stats()["policy"] == p.name


def test_simulation_helper_conserves_requests():
    """Guard on the test harness itself: a broken mock fleet would fake every result above."""
    keys = [f"k{i % 7}" for i in range(150)]
    out = simulate(LeastOutstanding(), keys)
    assert sum(out.counts.values()) == 150
    assert statistics.pstdev(list(out.counts.values())) < 1.0
