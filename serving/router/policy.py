"""
Routing policies: which replica gets this request.

THE ORGANIZING PRINCIPLE — THE ROUTER HOLDS HINTS, THE REPLICA HOLDS TRUTH
--------------------------------------------------------------------------
Every piece of router state in this file is an approximation that may be stale.
The prefix->replica table is a *hint* table: it records where a similar prefix
was sent last time, not where the KV blocks actually are. The load estimate is a
count of requests this router believes are in flight, not a reading of any
replica's scheduler. Both can be wrong at any instant, and neither is
coordinated with anything.

If a hint is wrong, the request lands on a suboptimal replica and runs slightly
slower — one prefix-cache miss, one extra prefill. **It never produces a wrong
answer, never corrupts cache state, and never needs a distributed transaction.**
That property is what makes this layer safe to build without consensus, a shared
cache tier, or a coordination service, and it is the first thing to say when
asked "how do you keep the router's view of the cache consistent?" — you don't,
and that's the design (docs/ARCHITECTURE.md §1, §6).

The corollary that shapes the code: nothing here is allowed to *fail* because a
hint is stale. `select()` returns a live, eligible replica or `None`; a hint
naming a quarantined, drained, or unknown replica is silently ignored rather
than raising, because the failure mode of a hint is degraded placement, and any
other failure mode would be a bug in this file rather than a property of the
system.

THE POLICIES, AND WHAT EACH ONE IS FOR (docs/BENCHMARK_METHODOLOGY.md §6)
-------------------------------------------------------------------------
`RoundRobin` (B4)
    Strict rotation. Independent of load, queue depth, and cache state; no
    re-routing after assignment. **A genuinely WEAK baseline.** Beating
    round-robin is table stakes, not a result, and any number quoted against it
    alone should be read as such.

`LeastOutstanding` (B5)
    Fewest in-flight requests, ties broken by rotation. **THE REAL BASELINE.**
    It is load-aware but cache-blind, so the delta between B5 and `PrefixAware`
    isolates *exactly* the value of cache awareness — which is the actual claim.
    A prefix-aware router that beats B4 but not B5 has demonstrated load
    balancing, not prefix awareness, and this module exists partly so that
    result is measurable and reportable as such.

`PrefixAware`
    An approximate, TTL'd prefix->replica hint table BLENDED with load. The
    blend is a tunable weight rather than a constant because methodology §10
    case 3 predicts, in advance, that affinity and load conflict near
    saturation: the replica holding the right prefix is often busy *because* it
    holds the popular prefix. See `PrefixAwareConfig` for the exact scoring
    function and what each knob buys.

`ConsistentHash`
    Optional, for comparison. Cache-affine without any state at all, and
    therefore load-blind in the strongest possible way — it cannot react to a
    hot prefix even in principle. Useful as the "what does affinity cost when it
    cannot adapt" column.

All four are behind one interface so the benchmark can swap them by name.
"""

from __future__ import annotations

import hashlib
import math
import struct
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ReplicaView",
    "RouteRequest",
    "PrefixKeyer",
    "HintTable",
    "RoutingPolicy",
    "RoundRobin",
    "LeastOutstanding",
    "PrefixAware",
    "PrefixAwareConfig",
    "ConsistentHash",
    "build_policy",
    "POLICIES",
]


# ---------------------------------------------------------------------------
# The view a policy gets of the fleet
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicaView:
    """
    One replica, as the policy sees it: a HINT, not a reading.

    `inflight` is the router's own count of requests it dispatched and has not
    yet seen finish. It is not the replica's queue depth — the replica may have
    admitted, batched, preempted, or shed since. It is close enough to order
    replicas by, which is all any policy here does with it.

    `weight` is the ramp-in capacity multiplier from `health.py` and is the one
    field that is not about the present at all: a recovered replica is *healthy*
    and *cold*, two different properties of which only one is health-checked
    (docs/ARCHITECTURE.md §9.3 step 5). Load is divided by it, so a replica at
    weight 0.1 has to be ten times less loaded before it looks equally
    attractive.
    """

    replica_id: str
    inflight: int = 0
    eligible: bool = True
    """Healthy AND accepting new work. Draining and quarantined replicas are not."""
    weight: float = 1.0
    """Ramp-in weight in (0, 1]. A capacity multiplier, not a probability."""
    warm: bool = True
    """False while ramping in. Purely informational for policies; `weight` does the work."""

    @property
    def effective_load(self) -> float:
        """In-flight scaled by ramp weight: a cold replica *looks* loaded, on purpose."""
        w = self.weight if self.weight > 0 else 1e-9
        return self.inflight / w


@dataclass(frozen=True)
class RouteRequest:
    """
    What a policy is allowed to know about a request.

    Deliberately tiny. `prefix_key` is an approximate, block-aligned digest of
    the head of the prompt (see `PrefixKeyer`) and may be `None` — a prompt too
    short to fill one block has nothing block-aligned to be affine about, and
    every policy must handle that by falling back to load.
    """

    request_id: str
    prefix_key: str | None = None
    prompt_tokens: int = 0
    """Approximate. Used for reporting only; no policy decision keys off it."""


# ---------------------------------------------------------------------------
# Prefix keying
# ---------------------------------------------------------------------------


def _digest(*parts: bytes) -> str:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p)
    return h.hexdigest()


@dataclass(frozen=True)
class PrefixKeyer:
    """
    Hash of the first N block-aligned tokens of a prompt.

    BLOCK ALIGNMENT IS THE WHOLE POINT. The replica's radix cache reuses whole
    blocks and only whole blocks: a prefix that matches 20 tokens at
    `block_size=16` yields ONE reusable block, not 20 reusable tokens
    (docs/ARCHITECTURE.md §4). A hint keyed on an unaligned token count would
    claim affinity for a match the replica cannot actually reuse, which is a
    hint that is wrong in the one direction that costs work rather than saving
    it. So the key is computed over `floor(n / block_size) * block_size` tokens.

    N BLOCKS, NOT THE WHOLE PROMPT. Keying on the entire prompt would make every
    request with a unique suffix a unique key — i.e. would turn the hint table
    into a per-request map with a 0% hit rate, which is exactly the workload
    shape (system-prompt sharing) the router exists to exploit. `n_blocks`
    bounds how much of the head has to match for two requests to be considered
    the same prefix. It is an approximation of the shared region, and being
    wrong about it costs locality, nothing else.

    THE ROUTER DOES NOT TOKENIZE, BY DEFAULT. Running a HuggingFace tokenizer
    per request on the router's single event loop is real CPU on the request
    path, and it buys exactness in a table whose entries are advisory anyway. So
    `key_from_text` approximates token counts as `chars / chars_per_token` and
    aligns in that approximate space. Two prompts sharing a long literal prefix
    still collide on the same key; the approximation only blurs where exactly
    the boundary falls. `key_from_tokens` is provided for callers that already
    have ids (the benchmark harness does) and is exact.
    """

    block_size: int = 16
    n_blocks: int = 4
    chars_per_token: float = 4.0

    @property
    def block_chars(self) -> int:
        return max(1, int(self.block_size * self.chars_per_token))

    def key_from_tokens(self, token_ids: Sequence[int]) -> str | None:
        n = min(len(token_ids), self.n_blocks * self.block_size)
        n -= n % self.block_size
        if n == 0:
            return None
        payload = struct.pack(f"<{n}i", *token_ids[:n])
        return _digest(b"tok", struct.pack("<ii", self.block_size, n), payload)

    def key_from_text(self, text: str) -> str | None:
        n = min(len(text), self.n_blocks * self.block_chars)
        n -= n % self.block_chars
        if n == 0:
            return None
        return _digest(b"txt", struct.pack("<ii", self.block_chars, n), text[:n].encode())

    def key_from_messages(self, messages: Iterable[Any]) -> str | None:
        """
        Key an OpenAI-style message list.

        Roles are included because a system message and a user message with the
        same text are different prompts once the chat template has run, and a
        key that conflated them would hand out an affinity the replica cannot
        honour.
        """
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                role, content = msg.get("role", ""), msg.get("content", "")
            else:
                role, content = getattr(msg, "role", ""), getattr(msg, "content", "")
            parts.append(f"{role}\x1f{content}")
        return self.key_from_text("\x1e".join(parts))


# ---------------------------------------------------------------------------
# The hint table
# ---------------------------------------------------------------------------


@dataclass
class HintTable:
    """
    prefix key -> the replica(s) that most recently served it. Lossy by design.

    THREE PROPERTIES, EACH FOR A REASON:

    * **TTL'd.** An entry older than `ttl_s` is treated as absent. Replicas
      evict (LRU over the radix trie), so a hint's truth decays on its own with
      no event this router will ever see. A TTL is the cheapest possible
      approximation of that decay, and methodology §10 case 5 (high cache
      turnover) is exactly the regime where a too-long TTL makes the signal
      actively misleading — "route to the replica that had it" finds it gone.

    * **Bounded.** `max_entries` caps memory; the oldest entries are dropped
      first. Dropping a hint is always safe (see the module docstring), so the
      eviction policy needs no more sophistication than that.

    * **Optionally multi-valued.** `fanout > 1` lets one key name several
      replicas, which is the direct mitigation for §10 case 7: a single very hot
      prefix routed by strict affinity is a self-inflicted hotspot. Replicating
      the hint spreads the hot key over `fanout` replicas at the cost of
      `fanout`x the prefill for that prefix. Default 1 — the mitigation is
      opt-in so the un-mitigated hotspot is measurable.
    """

    ttl_s: float = 60.0
    max_entries: int = 100_000
    fanout: int = 1
    _entries: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    expired: int = 0
    purged: int = 0

    def get(self, key: str | None, now: float) -> list[str]:
        if key is None:
            return []
        entry = self._entries.get(key)
        if not entry:
            self.misses += 1
            return []
        live = [(rid, ts) for rid, ts in entry if now - ts <= self.ttl_s]
        if len(live) != len(entry):
            self.expired += len(entry) - len(live)
            if live:
                self._entries[key] = live
            else:
                self._entries.pop(key, None)
        if not live:
            self.misses += 1
            return []
        self.hits += 1
        return [rid for rid, _ in live]

    def put(self, key: str | None, replica_id: str, now: float) -> None:
        if key is None:
            return
        entry = [(rid, ts) for rid, ts in self._entries.get(key, ()) if rid != replica_id]
        entry.insert(0, (replica_id, now))
        self._entries[key] = entry[: self.fanout]
        if len(self._entries) > self.max_entries:
            self._evict_oldest()

    def purge_replica(self, replica_id: str) -> int:
        """
        Drop every hint naming this replica. Called the instant it is quarantined.

        This is step 2 of docs/ARCHITECTURE.md §9.3 and it is not housekeeping:
        the replica's KV cache died with the process, so every hint pointing at
        it is now not merely stale but *anti-informative* — it steers traffic
        toward a dead replica's prefixes, and would keep doing so after the
        replica restarts cold, sending exactly the requests that expect a warm
        cache to the one replica guaranteed not to have one.
        """
        removed = 0
        for key in list(self._entries):
            entry = [(rid, ts) for rid, ts in self._entries[key] if rid != replica_id]
            removed += len(self._entries[key]) - len(entry)
            if entry:
                self._entries[key] = entry
            else:
                del self._entries[key]
        self.purged += removed
        return removed

    def _evict_oldest(self) -> None:
        drop = max(1, len(self._entries) // 10)
        by_age = sorted(self._entries.items(), key=lambda kv: max(ts for _, ts in kv[1]))
        for key, _ in by_age[:drop]:
            del self._entries[key]

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "ttl_s": self.ttl_s,
            "fanout": self.fanout,
            "hint_hits": self.hits,
            "hint_misses": self.misses,
            "hint_hit_rate": (self.hits / total) if total else None,
            "entries_expired_total": self.expired,
            "entries_purged_total": self.purged,
            "note": (
                "A hint hit is NOT a cache hit. It means the router believed a "
                "replica held this prefix; only the replica's own block-granularity "
                "hit rate says whether it did (BENCHMARK_METHODOLOGY §7)."
            ),
        }


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------


class RoutingPolicy(ABC):
    """
    One method does the work; the rest is bookkeeping the policy may ignore.

    Every implementation must satisfy three properties, and the tests assert all
    three because they are what let the router be swapped mid-benchmark:

    1. `select` returns an eligible replica id or `None`. `None` means "shed" —
       never an exception, never an ineligible replica.
    2. Given the same construction arguments and the same call sequence, the
       selections are identical. No policy reads a clock it was not given, and
       no policy calls `random` without a seeded generator.
    3. Losing every scrap of internal state (hints, cursors) degrades placement
       and nothing else.
    """

    name: str = "abstract"

    @abstractmethod
    def select(self, req: RouteRequest, replicas: Sequence[ReplicaView], now: float) -> str | None:
        """Choose a replica for `req`, or `None` to shed."""

    def on_assign(self, req: RouteRequest, replica_id: str, now: float) -> None:  # noqa: B027
        """
        The request was dispatched. Default: NOTHING, and deliberately not abstract.

        A stateless policy (B4, B5, consistent hashing) has nothing to learn from
        a dispatch. Forcing all three to implement an empty override would be
        ceremony; `PrefixAware` is the only policy with a table to update.
        """

    def on_complete(  # noqa: B027 — optional hook, see on_assign
        self, req: RouteRequest, replica_id: str, now: float, *, ok: bool = True
    ) -> None:
        """The request finished (or failed). Default: nothing."""

    def purge_replica(self, replica_id: str) -> None:  # noqa: B027 — see on_assign
        """
        The replica died. Default: nothing to forget, because a stateless policy
        holds no belief about it that could have gone stale.
        """

    def stats(self) -> dict[str, Any]:
        return {"policy": self.name}

    # -- shared helpers -----------------------------------------------------

    @staticmethod
    def _eligible(replicas: Sequence[ReplicaView]) -> list[ReplicaView]:
        return [r for r in replicas if r.eligible]


class _RotatingPolicy(RoutingPolicy):
    """Shared rotation cursor, so 'ties broken by rotation' means one thing everywhere."""

    def __init__(self) -> None:
        self._cursor = 0

    def _rotate(self, ordered: Sequence[ReplicaView], candidates: Sequence[str]) -> str:
        """
        Pick the first candidate at or after the cursor in fleet order, wrapping.

        Fleet order — not candidate order — is what makes this a rotation rather
        than "always the first tied replica": the cursor advances past the chosen
        replica's position in the full list, so the next tie starts looking
        somewhere else.
        """
        allowed = set(candidates)
        n = len(ordered)
        for offset in range(n):
            ix = (self._cursor + offset) % n
            if ordered[ix].replica_id in allowed:
                self._cursor = (ix + 1) % n
                return ordered[ix].replica_id
        return candidates[0]  # unreachable while candidates ⊆ ordered


# ---------------------------------------------------------------------------
# B4 — round robin
# ---------------------------------------------------------------------------


class RoundRobin(_RotatingPolicy):
    """
    B4. Strict rotation, and DELIBERATELY WEAK.

    Load, queue depth, and cache state are all invisible to it, and there is no
    re-routing after assignment (methodology §6/B4). It will happily hand the
    next request to a replica with thirty outstanding requests while another
    sits idle, because rotation is the entire policy.

    **Beating this is table stakes, not a result.** It is here because it is the
    thing everybody's first router does, and because the B4-to-B5 gap is the
    honest measure of how much of any improvement is just load balancing.

    The one concession to reality: an ineligible replica is skipped rather than
    dispatched to. That is not load awareness — it is the difference between a
    routing policy and a null pointer.
    """

    name = "round_robin"

    def select(self, req: RouteRequest, replicas: Sequence[ReplicaView], now: float) -> str | None:
        eligible = self._eligible(replicas)
        if not eligible:
            return None
        return self._rotate(replicas, [r.replica_id for r in eligible])


# ---------------------------------------------------------------------------
# B5 — least outstanding
# ---------------------------------------------------------------------------


class LeastOutstanding(_RotatingPolicy):
    """
    B5. Fewest in-flight requests; ties broken by rotation. **THE REAL BASELINE.**

    Load-aware and cache-blind. That combination is the whole reason it is the
    baseline that matters: it already captures every benefit that comes from not
    piling work onto a busy replica, so whatever `PrefixAware` gains over it is
    attributable to cache awareness and to nothing else (methodology §6/B5).

    Ramp weight is honoured through `effective_load`, so B5 does not send a
    cold, just-recovered replica the same share as a warm one. That is a
    deviation from the textbook definition and it is deliberate: without it, B5
    and the prefix-aware policy would differ in *two* respects during a
    fault-injection run — cache awareness and recovery handling — and the
    comparison would no longer isolate the claim.
    """

    name = "least_outstanding"

    def select(self, req: RouteRequest, replicas: Sequence[ReplicaView], now: float) -> str | None:
        eligible = self._eligible(replicas)
        if not eligible:
            return None
        best = min(r.effective_load for r in eligible)
        tied = [r.replica_id for r in eligible if math.isclose(r.effective_load, best)]
        return self._rotate(replicas, tied)


# ---------------------------------------------------------------------------
# Prefix-aware
# ---------------------------------------------------------------------------


@dataclass
class PrefixAwareConfig:
    """
    The affinity/load blend, and every knob that shifts it.

    THE SCORING FUNCTION, in full:

        load_norm(r) = min(1, effective_load(r) / load_scale)
        affinity(r)  = 1 if r is hinted for this prefix AND
                            effective_load(r) < saturation_inflight, else 0
        score(r)     = blend * affinity(r) - (1 - blend) * load_norm(r)

    and the highest score wins, ties broken by the same rotation B5 uses.

    WHY IT IS SHAPED LIKE THIS. Methodology §10 case 3 predicts, before any
    measurement, that affinity and load conflict *directly* near saturation: the
    replica holding the right prefix is frequently the busy one precisely
    BECAUSE it holds the popular prefix. Routing for affinity concentrates load;
    routing for balance destroys affinity. A policy with no knob between those
    two is a policy that must lose one of the two regimes.

    * `blend` = 1.0 -> pure affinity. Cache-optimal, load-blind, and the
      configuration §10 cases 2, 3 and 7 predict will lose.
    * `blend` = 0.0 -> exactly B5. Asserted by a test, because a blend parameter
      whose zero is not the baseline is a parameter nobody can interpret.
    * In between, one prefill's worth of saved work is being traded against one
      unit of normalized queueing delay. The exchange rate is workload- and
      hardware-dependent — it depends on where the knee is — which is why this
      is a swept parameter and not a constant (docs/ARCHITECTURE.md §11).

    `saturation_inflight` is a HARD floor under that trade rather than more
    curve. Past it, affinity is simply not offered: no prefill saving repays
    queueing behind a saturated replica, and a purely linear blend would keep
    paying it at a slightly worse exchange rate forever. Set it to `None` to
    disable the cutoff and measure the un-mitigated §10 case 3 loss.

    `load_scale` is the in-flight count at which a replica counts as fully
    loaded for scoring. It defaults to `saturation_inflight` so that the two
    knobs cannot drift apart silently.
    """

    blend: float = 0.7
    saturation_inflight: float | None = 8.0
    load_scale: float | None = None
    hint_ttl_s: float = 60.0
    hint_fanout: int = 1
    max_entries: int = 100_000
    keyer: PrefixKeyer = field(default_factory=PrefixKeyer)

    def resolved_load_scale(self) -> float:
        if self.load_scale is not None:
            return max(1e-9, self.load_scale)
        if self.saturation_inflight is not None:
            return max(1e-9, self.saturation_inflight)
        return 8.0


class PrefixAware(_RotatingPolicy):
    """
    Approximate, TTL'd prefix->replica hints, blended with load.

    The hint is written on ASSIGNMENT, not on completion, and that is the
    self-correcting part: whatever replica the blend actually chose becomes the
    hint, so a request diverted away from a saturated affine replica moves the
    prefix's hint with it rather than leaving the table pointing at a replica the
    policy has decided to stop using.

    Nothing in here can produce a wrong answer. A hint naming a replica that has
    since been quarantined, drained, or restarted cold is filtered out by
    eligibility and by `purge_replica`; a hint that is simply wrong (the replica
    evicted that prefix ten seconds ago) costs one prefill. See the module
    docstring.
    """

    name = "prefix_aware"

    def __init__(self, config: PrefixAwareConfig | None = None) -> None:
        super().__init__()
        self.config = config or PrefixAwareConfig()
        self.hints = HintTable(
            ttl_s=self.config.hint_ttl_s,
            max_entries=self.config.max_entries,
            fanout=self.config.hint_fanout,
        )
        self.affinity_honoured = 0
        self.affinity_declined_saturated = 0
        self.affinity_declined_load = 0
        self.no_hint = 0

    # -- selection ----------------------------------------------------------

    def select(self, req: RouteRequest, replicas: Sequence[ReplicaView], now: float) -> str | None:
        eligible = self._eligible(replicas)
        if not eligible:
            return None

        hinted_all = set(self.hints.get(req.prefix_key, now))
        hinted = {r.replica_id for r in eligible if r.replica_id in hinted_all}
        if not hinted:
            self.no_hint += 1

        cfg = self.config
        scale = cfg.resolved_load_scale()
        sat = cfg.saturation_inflight

        saturated_hint = False
        scores: dict[str, float] = {}
        for r in eligible:
            load_norm = min(1.0, r.effective_load / scale)
            affine = r.replica_id in hinted
            if affine and sat is not None and r.effective_load >= sat:
                # §10 case 3, made explicit: past saturation the prefill saving
                # cannot repay the queueing, so affinity is not offered at all.
                affine = False
                saturated_hint = True
            aff = 1.0 if affine else 0.0
            scores[r.replica_id] = cfg.blend * aff - (1.0 - cfg.blend) * load_norm

        best = max(scores.values())
        tied = [rid for rid, s in scores.items() if math.isclose(s, best, abs_tol=1e-12)]
        chosen = self._rotate(replicas, tied)

        if hinted:
            if chosen in hinted:
                self.affinity_honoured += 1
            elif saturated_hint:
                self.affinity_declined_saturated += 1
            else:
                self.affinity_declined_load += 1
        return chosen

    # -- bookkeeping --------------------------------------------------------

    def on_assign(self, req: RouteRequest, replica_id: str, now: float) -> None:
        self.hints.put(req.prefix_key, replica_id, now)

    def purge_replica(self, replica_id: str) -> None:
        self.hints.purge_replica(replica_id)

    def stats(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "blend": self.config.blend,
            "saturation_inflight": self.config.saturation_inflight,
            "affinity_honoured": self.affinity_honoured,
            "affinity_declined_saturated": self.affinity_declined_saturated,
            "affinity_declined_load": self.affinity_declined_load,
            "requests_without_usable_hint": self.no_hint,
            "hints": self.hints.stats(),
        }


# ---------------------------------------------------------------------------
# Consistent hashing — the comparison arm
# ---------------------------------------------------------------------------


class ConsistentHash(RoutingPolicy):
    """
    Ring hashing over replica ids. Stateless affinity, and load-blind by construction.

    Included for comparison, not as a candidate. It gets prefix affinity for
    free and with zero bookkeeping — no table, no TTL, no purge — but it cannot
    react to load *even in principle*, so §10 case 7 (one very hot prefix) has
    no mitigation available: every request for that key goes to one replica
    until the ring itself changes. That makes it the clean control for "how much
    of the prefix-aware result comes from adapting to load rather than from
    affinity."

    A replica leaving the ring moves only its own keys (the point of consistent
    hashing), which matters here because a quarantine that reshuffled every key
    would invalidate every replica's cache at once — turning one replica's death
    into a fleet-wide cold start.
    """

    name = "consistent_hash"

    def __init__(self, virtual_nodes: int = 128) -> None:
        self.virtual_nodes = virtual_nodes
        self._ring: list[tuple[int, str]] = []
        self._ring_members: tuple[str, ...] = ()
        self._fallback = LeastOutstanding()

    def _ensure_ring(self, members: Sequence[str]) -> None:
        key = tuple(sorted(members))
        if key == self._ring_members:
            return
        ring: list[tuple[int, str]] = []
        for rid in key:
            for v in range(self.virtual_nodes):
                point = int(_digest(rid.encode(), b"#", str(v).encode()), 16)
                ring.append((point, rid))
        ring.sort()
        self._ring = ring
        self._ring_members = key

    def select(self, req: RouteRequest, replicas: Sequence[ReplicaView], now: float) -> str | None:
        eligible = self._eligible(replicas)
        if not eligible:
            return None
        if req.prefix_key is None:
            # Nothing block-aligned to hash. Falling back to load is strictly
            # better than hashing the request id, which would be a random pick
            # dressed up as a policy.
            return self._fallback.select(req, replicas, now)
        self._ensure_ring([r.replica_id for r in eligible])
        point = int(_digest(req.prefix_key.encode()), 16)
        lo, hi = 0, len(self._ring)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._ring[mid][0] < point:
                lo = mid + 1
            else:
                hi = mid
        return self._ring[lo % len(self._ring)][1]

    def stats(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "virtual_nodes": self.virtual_nodes,
            "ring_members": list(self._ring_members),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

POLICIES = {
    RoundRobin.name: RoundRobin,
    LeastOutstanding.name: LeastOutstanding,
    PrefixAware.name: PrefixAware,
    ConsistentHash.name: ConsistentHash,
}


def build_policy(name: str, **kwargs: Any) -> RoutingPolicy:
    """
    Construct a policy by name so the benchmark can sweep it from a flag.

    Unknown names raise here rather than silently falling back to a default: a
    benchmark that ran B5 while its artifact said `prefix_aware` is the
    worst-shaped error this project can make, and it is the same class of
    mistake `SERVING_STATIC_BATCHING` prints a loud banner about.
    """
    if name not in POLICIES:
        raise KeyError(f"unknown routing policy {name!r}; known: {sorted(POLICIES)}")
    cls = POLICIES[name]
    if cls is PrefixAware and kwargs:
        return PrefixAware(PrefixAwareConfig(**kwargs))
    return cls(**kwargs)
