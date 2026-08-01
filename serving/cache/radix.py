"""Radix prefix cache: token trie, refcounting, LRU eviction, copy-on-write.

AUTHOR-WRITTEN. Phase 4.
Explainability gate: block-boundary truncation, why refcount AND LRU are both
needed, when COW triggers.

WHAT THIS IS
------------
A trie over token ids in which **one edge spans exactly one block's worth of
tokens** and each node holds:

  * the physical block id whose KV covers those tokens,
  * a reference *held through the allocator* (never a private counter),
  * a last-access timestamp for LRU.

Lookup walks the incoming prompt down the trie and stops at the deepest matching
node; the matched blocks are increfed into the new sequence and the remainder is
fresh prefill work (docs/ARCHITECTURE.md §4, §9.1).

GRANULARITY IS THE BLOCK, NOT THE TOKEN — AND THIS IS THE WHOLE BUG SURFACE
---------------------------------------------------------------------------
A prefix that matches 20 tokens at `block_size=16` yields **one** reusable block,
not 20 reusable tokens. Reuse is valid only for a block whose *every* token
matches, because the block is the unit of KV the attention kernel dereferences:
there is no way to say "attend to the first four positions of this page and
recompute the rest". If divergence falls mid-block, that block is recomputed.

That truncation is the §9.1 case and it is R6's exact hazard. A cache that
reused the straddling block would be **faster and wrong**: every throughput and
TTFT number improves, the output stays fluent, and nothing raises. So the walk
below only ever consults *whole* blocks (`_full_block_keys`), and the trie
literally cannot represent a partial block — a partial match is unrepresentable
rather than merely unwritten.

THE ALLOCATOR OWNS REFCOUNTS. THIS MODULE DOES NOT.
---------------------------------------------------
`serving/memory/allocator.py` is explicit that two components which can both
decide a block is free will eventually disagree, and that the failure is silent
(R7). So this cache never keeps a private count. Instead:

    **the cache holds exactly ONE allocator reference per resident node.**

Consequences worth stating, because everything else follows from them:

  * A cached block's refcount is `1 + (number of live sequences using it)`.
    Residency and use are therefore distinguishable with no extra state.
  * A node is *evictable* exactly when `allocator.refcount(block) == 1` — the
    cache is the only holder. That is what `_users()` computes, and it is why
    eviction asks the allocator rather than trusting LRU position.
  * A retiring sequence needs no `release()` call into this module.
    `SequenceBlocks.free()` decrefs every block it holds; a shared block simply
    drops from 2 to 1 and stays resident. One code path, no second owner.

REFCOUNT AND LRU ARE BOTH NEEDED, AND THEY ARE NOT THE SAME QUESTION
--------------------------------------------------------------------
LRU answers "which block is least valuable to keep?". Refcount answers "which
block is *legal* to drop?". A block can be simultaneously the least recently
accessed node in the trie and be live inside three running sequences — LRU
position says evict, refcount says absolutely not, and refcount wins. Eviction
that consulted only LRU order would hand a live sequence's KV to a different
sequence, which produces fluent, wrong output and moves no metric (R7).

And the converse asymmetry, which is the subtle half:

    a block whose refcount drops back to zero re-enters eviction eligibility
    with its ORIGINAL access time, not a refreshed one.

`last_access` is stamped on *insert* and on a *lookup hit* — both are genuine
accesses, evidence that somebody wanted this prefix. It is deliberately **not**
stamped when a sequence retires. Retirement is the opposite of interest. If
release refreshed the timestamp, a single long-running request would keep
bumping its prefix to the head of the LRU list for the entire duration of its
generation, so a prefix nobody else ever asked for would outlive prefixes that
are being hit constantly. The long request would be protecting the cache entry
*from* the workload rather than the workload protecting it. `release` therefore
does not exist here at all, which makes the property unbreakable rather than
merely observed (`test_refcount_return_to_zero_keeps_original_access_time`).

LEAF-FIRST EVICTION
-------------------
An internal node cannot be evicted while a descendant exists. Its block is part
of that descendant's prefix: dropping it would leave a trie path whose earlier
blocks are gone, so a later walk would match the deep node and hand out a block
table with a hole in the middle of the sequence. Eviction therefore only ever
considers leaves, and evicting a leaf may expose its parent as the next
candidate — which is why `evict()` loops rather than collecting a batch up
front.

COPY-ON-WRITE, AND WHY THE PREFILL PATH NEVER TRIGGERS IT
----------------------------------------------------------
COW applies to exactly one block: the **last, partially-filled** block of a
shared prefix. Fully shared blocks are never written to — they are full, by
definition, so there is nothing left to write into them.

Under strict block-granularity reuse this is a *guard*, not a hot path, and that
is a real result rather than an omission. `acquire()` returns a whole number of
full blocks, so a reusing sequence's first write lands at token offset
`n_matched_blocks * block_size` — the first slot of a **freshly allocated**
block. It structurally cannot write into a block another sequence is reading.
`ensure_writable()` exists because that invariant must be *enforced and tested*
rather than assumed, and because any future extension (partial-block reuse,
beam-search forks, parallel sampling off a common prompt) breaks it immediately.
It refuses to COW anything but a sequence's final block, so "a full shared block
is never written" is checked rather than hoped for.

COST WHEN IT NEVER HELPS
------------------------
Lookup and insert are O(prompt_blocks) dict operations on tuples of `block_size`
ints; eviction is an O(resident nodes) scan that runs only under memory
pressure. The zero-sharing control (methodology §4, §7) exists to publish that
cost rather than assume it away, and `tests/test_radix.py` bounds it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from serving.memory.allocator import AllocationError

if TYPE_CHECKING:
    from serving.memory.allocator import BlockAllocator
    from serving.memory.block_table import SequenceBlocks

__all__ = [
    "CacheStats",
    "MatchResult",
    "RadixCache",
    "RadixNode",
    "attach_prefix",
]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class RadixNode:
    """
    One block's worth of tokens, and the physical block holding their KV.

    `eq=False` so nodes hash by identity: two distinct nodes can legitimately
    carry the same `key` tuple at different depths (the same 16 tokens appearing
    after different prefixes are different KV entirely, because attention is
    causal over everything before them). Value equality would silently conflate
    them, which is the one confusion that would produce cross-prefix KV reuse.

    `last_access` is a logical clock, not wall time — the ordering is all that
    matters and a monotonic counter cannot tie or go backwards under a clock
    adjustment.
    """

    key: tuple[int, ...] = ()
    block_id: int = -1
    parent: RadixNode | None = None
    children: dict[tuple[int, ...], RadixNode] = field(default_factory=dict)
    last_access: int = 0
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"RadixNode(depth={self.depth}, block={self.block_id}, "
            f"children={len(self.children)}, last_access={self.last_access})"
        )


@dataclass
class MatchResult:
    """
    The outcome of one prefix walk, in the units methodology §7 reports.

    `n_tokens` is always `len(block_ids) * block_size` — a whole number of
    blocks, never a token count that happens to fall mid-block. If those two
    ever disagree, the block-granularity property has been broken.
    """

    block_ids: list[int] = field(default_factory=list)
    n_tokens: int = 0
    depth: int = 0
    blocks_required: int = 0
    truncated_partial_block: bool = False
    """
    True when the walk stopped because divergence fell INSIDE a block that
    otherwise shared a prefix — the §9.1 case. Recorded because it is the
    situation R6 lives in, and a cache that never reports it on the adversarial
    workload is not doing block-granularity matching.
    """

    truncated_tokens: int = 0
    """
    Tokens that genuinely matched inside the straddling block and were still
    recomputed. This is the price of block granularity, in tokens, and it is
    bounded by `block_size - 1`. Publishing it is what stops "our hit rate is
    only 60%" from being mistaken for a matching bug.
    """

    @property
    def hit(self) -> bool:
        return bool(self.block_ids)

    @property
    def hit_rate(self) -> float:
        """blocks_reused / blocks_required_at_prefill (methodology §7)."""
        return (len(self.block_ids) / self.blocks_required) if self.blocks_required else 0.0


@dataclass
class CacheStats:
    """
    Block-granularity instrumentation (docs/BENCHMARK_METHODOLOGY.md §7).

    Hit rate is `blocks_reused / blocks_required_for_prefill` because that is
    the granularity at which the allocator actually saves work. A token-level
    rate would flatter the cache by counting the partially matched block that
    §9.1 forces us to recompute.
    """

    lookups: int = 0
    hits: int = 0
    blocks_reused: int = 0
    blocks_required: int = 0
    partial_block_truncations: int = 0
    truncated_tokens: int = 0
    inserts: int = 0
    evictions: int = 0
    cow_copies: int = 0
    partial_hit_depths: list[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return (self.blocks_reused / self.blocks_required) if self.blocks_required else 0.0

    @property
    def request_hit_rate(self) -> float:
        return (self.hits / self.lookups) if self.lookups else 0.0

    @property
    def mean_partial_hit_depth(self) -> float:
        d = self.partial_hit_depths
        return (sum(d) / len(d)) if d else 0.0


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


class RadixCache:
    """
    Block-granularity radix prefix cache over a `BlockAllocator`.

    Not thread-safe, matching the allocator and the scheduler: one scheduler,
    one allocator, one cache, one event loop (docs/ARCHITECTURE.md §7).
    """

    def __init__(
        self,
        allocator: BlockAllocator,
        *,
        max_cached_blocks: int | None = None,
        block_copy: Callable[[int, int], None] | None = None,
        enabled: bool = True,
    ):
        """
        Parameters
        ----------
        max_cached_blocks
            Residency budget. The cache holds a real allocator reference per
            node, so an unbounded cache can consume the pool and starve a
            RUNNING sequence's growth — and growth deliberately bypasses the
            watermark (`allocator.allocate`, not `can_allocate`), so the
            watermark will not save it. `None` means "evict only on demand",
            which is correct when the caller drives `reserve()` at admission and
            wrong if nobody does. A server should set it.
        block_copy
            `(src_block_id, dst_block_id) -> None`, supplied by the attention
            backend, used by copy-on-write. `None` means COW is structurally
            available but cannot physically copy — fine for CPU tests and for a
            configuration that never hits COW, and `ensure_writable()` raises
            rather than silently repointing to an uninitialised block.
        enabled
            The A/B switch. `False` makes every lookup a miss and every insert a
            no-op, so cache-on-vs-cache-off equality (R6's detection) is a
            single flag over an otherwise identical stack.
        """
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.enabled = enabled
        self.max_cached_blocks = max_cached_blocks
        self._block_copy = block_copy

        self.root = RadixNode()
        self._nodes: set[RadixNode] = set()          # every node except the root
        self._by_block: dict[int, RadixNode] = {}    # physical block -> node, 1:1
        self._clock = 0

        self.stats = CacheStats()

    # -- clock --------------------------------------------------------------

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    # -- introspection ------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def cached_blocks(self) -> int:
        """Blocks held resident by the cache itself."""
        return len(self._by_block)

    @property
    def max_depth(self) -> int:
        return max((n.depth for n in self._nodes), default=0)

    @property
    def shared_blocks(self) -> int:
        """
        Cached blocks with `refcount > 1` — i.e. GENUINELY shared right now, held
        by the cache AND by at least one live sequence. Methodology §7 asks for
        this separately from hit rate because a high hit rate with zero
        concurrently-shared blocks means the workload is sequential reuse, not
        concurrent sharing, and those two make very different capacity claims.
        """
        return sum(1 for b in self._by_block if self.allocator.refcount(b) > 1)

    def _users(self, node: RadixNode) -> int:
        """
        Live sequences holding this node's block.

        `refcount - 1` because the cache itself always holds exactly one
        reference for a resident node. This is the number eviction consults, and
        it is asked of the ALLOCATOR rather than tracked here — one owner.
        """
        return self.allocator.refcount(node.block_id) - 1

    # -- the walk -----------------------------------------------------------

    def _full_block_keys(self, token_ids: Sequence[int]) -> list[tuple[int, ...]]:
        """
        Split into WHOLE blocks, discarding any ragged tail.

        The discard is the block-granularity property in one line: a trailing
        partial block is not a key, cannot be inserted, and cannot be matched.
        Everything §9.1 warns about is prevented here rather than handled later.
        """
        bs = self.block_size
        n_full = len(token_ids) // bs
        return [tuple(token_ids[i * bs : (i + 1) * bs]) for i in range(n_full)]

    def match(self, token_ids: Sequence[int]) -> MatchResult:
        """
        Walk `token_ids` down the trie. PURE — no refcounts, no timestamps.

        Exists separately from `acquire()` so admission can ask "how many blocks
        would this request actually need?" before committing to anything. A
        probe that increfed would have to be unwound on the (common) path where
        admission then declines, and unwinding refcounts on an error path is
        precisely the code that gets skipped once and corrupts memory silently.
        """
        bs = self.block_size
        required = (len(token_ids) + bs - 1) // bs
        result = MatchResult(blocks_required=required)
        if not self.enabled or not token_ids:
            return result

        keys = self._full_block_keys(token_ids)
        node = self.root
        matched: list[RadixNode] = []
        for k in keys:
            child = node.children.get(k)
            if child is None:
                break
            matched.append(child)
            node = child

        # DID THE MATCH DIE INSIDE A BLOCK? Compare the block that failed
        # against the siblings offered at this node: a shared leading run means
        # the two prompts agree for part of that block and diverge partway
        # through it. Those agreeing tokens are real sharing that the cache
        # CANNOT sell, because reuse is only valid for a fully matching block —
        # the §9.1 case, measured rather than assumed.
        if len(matched) < len(keys):
            failed = keys[len(matched)]
            best = 0
            for sib in node.children:
                common = 0
                for a, b in zip(sib, failed, strict=True):
                    if a != b:
                        break
                    common += 1
                best = max(best, common)
            if best:
                result.truncated_partial_block = True
                result.truncated_tokens = best

        # THE LAST-TOKEN RULE. A request must contribute at least one token to
        # the forward pass or there is nothing to compute logits from: a batch
        # entry with query_len == 0 is rejected by build_batch_meta, and rightly
        # so. If the ENTIRE prompt is cached, hand back one block less and
        # recompute its tokens. Costs one block of prefill on an exact repeat;
        # the alternative is a request that can never produce a token.
        while matched and len(matched) * bs >= len(token_ids):
            matched.pop()

        result.block_ids = [n.block_id for n in matched]
        result.depth = len(matched)
        result.n_tokens = len(matched) * bs
        assert result.n_tokens % bs == 0, "match must return whole blocks — §9.1"
        return result

    def acquire(self, token_ids: Sequence[int]) -> MatchResult:
        """
        Match, then claim a reference on every matched block and refresh its LRU
        position. The caller owns those references from here and MUST release
        them (`SequenceBlocks.free()` does, via the allocator).

        Timestamps are refreshed here because a lookup hit is real evidence that
        the workload wants this prefix. They are NOT refreshed on release — see
        the module docstring.
        """
        res = self.match(token_ids)
        self.stats.lookups += 1
        self.stats.blocks_required += res.blocks_required
        if res.truncated_partial_block:
            self.stats.partial_block_truncations += 1
            self.stats.truncated_tokens += res.truncated_tokens
        if not res.block_ids:
            self.stats.partial_hit_depths.append(0)
            return res

        self.allocator.incref(res.block_ids)
        now = self._tick()
        for bid in res.block_ids:
            self._by_block[bid].last_access = now

        self.stats.hits += 1
        self.stats.blocks_reused += len(res.block_ids)
        self.stats.partial_hit_depths.append(res.depth)
        return res

    # -- insert -------------------------------------------------------------

    def insert(self, token_ids: Sequence[int], block_ids: Sequence[int]) -> int:
        """
        Publish computed blocks into the trie. Returns the number of NEW nodes.

        `token_ids` must be exactly the tokens whose KV is already computed and
        resident in `block_ids` — prompt tokens after prefill, or prompt plus
        generated tokens at retirement (which is what makes multi-turn
        conversational reuse work at all).

        Only WHOLE blocks are published. The sequence's trailing partial block
        is deliberately left out: its KV is incomplete, and a later reader
        matching it would attend over slots that were never written. That is the
        same off-by-one as §9.1 seen from the writing side.

        Insertion is idempotent per path. If a node already exists, the existing
        block wins and the caller's block is left alone — it will be freed with
        the sequence. Two sequences that raced to compute the same prefix
        produce identical KV, so which physical copy survives is arbitrary; what
        must not happen is two nodes claiming the same path or one block being
        cached at two paths, both of which would break the 1:1 block-to-node map
        that `_users()` depends on.
        """
        if not self.enabled:
            return 0

        bs = self.block_size
        n_full = min(len(token_ids) // bs, len(block_ids))
        node = self.root
        inserted = 0

        for i in range(n_full):
            key = tuple(token_ids[i * bs : (i + 1) * bs])
            child = node.children.get(key)
            if child is not None:
                child.last_access = self._tick()
                node = child
                continue

            bid = block_ids[i]
            if bid in self._by_block:
                # This physical block is already cached at another path. Cannot
                # be true for a correct caller (a block belongs to one sequence
                # at one offset), so stop rather than corrupt the map.
                break
            try:
                self.allocator.incref([bid])
            except AllocationError:
                # Block is free — the sequence was already retired, or the
                # caller handed us a stale table. Nothing to cache, and forcing
                # it would resurrect a block whose contents are not guaranteed.
                break

            child = RadixNode(
                key=key,
                block_id=bid,
                parent=node,
                depth=node.depth + 1,
                last_access=self._tick(),
            )
            node.children[key] = child
            self._nodes.add(child)
            self._by_block[bid] = child
            inserted += 1
            node = child

        self.stats.inserts += inserted
        self._enforce_budget()
        return inserted

    # -- eviction -----------------------------------------------------------

    def _remove(self, node: RadixNode) -> None:
        """Drop one LEAF node and release the cache's reference to its block."""
        assert node.is_leaf, (
            f"attempted to evict internal node at depth {node.depth} with "
            f"{len(node.children)} children — its block is part of a live descendant's "
            "prefix, and removing it would leave a hole mid-sequence"
        )
        assert self._users(node) == 0, (
            f"attempted to evict block {node.block_id} with {self._users(node)} live "
            "users. This is R7: the victim sequence's attention would silently read "
            "another sequence's KV."
        )
        parent = node.parent
        if parent is not None:
            parent.children.pop(node.key, None)
        self._nodes.discard(node)
        self._by_block.pop(node.block_id, None)
        self.allocator.free([node.block_id])
        self.stats.evictions += 1

    def _evict_one(self) -> bool:
        """
        Evict the least-recently-accessed EVICTABLE leaf. Returns False when
        nothing is evictable.

        Two filters, and they answer different questions (module docstring):
        `is_leaf` is legality with respect to the trie, `_users() == 0` is
        legality with respect to memory. LRU only breaks ties among the blocks
        that both filters already allow.

        Linear scan over resident nodes. Deliberate: eviction only runs under
        memory pressure, and an intrusive LRU list would need updating on every
        lookup hit — paid on the hot path to speed up the cold one. If a profile
        ever says otherwise, the fix is a leaf-only heap, not a weaker filter.
        """
        victim: RadixNode | None = None
        for n in self._nodes:
            if n.children:
                continue
            if self._users(n) > 0:
                continue
            if victim is None or n.last_access < victim.last_access:
                victim = n
        if victim is None:
            return False
        self._remove(victim)
        return True

    def evict(self, n_blocks: int) -> int:
        """
        Free up to `n_blocks` cached blocks, least-recently-accessed first.

        Loops one at a time rather than collecting a batch: evicting a leaf can
        expose its parent as the next-best candidate, and a batch chosen up
        front would never see it.
        """
        if n_blocks <= 0:
            return 0
        freed = 0
        while freed < n_blocks and self._evict_one():
            freed += 1
        return freed

    def reserve(self, n_blocks: int, *, watermark: bool = True) -> bool:
        """
        Make room for `n_blocks`, evicting cached-but-unused blocks if needed.

        Returns whether the allocation can now proceed. `watermark=True` asks
        the admission question (`can_allocate`); a running sequence growing by
        one block asks with `watermark=False`, because the watermark exists to
        stop admission, not to starve a sequence that was already admitted.
        """
        def ok() -> bool:
            if watermark:
                return self.allocator.can_allocate(n_blocks)
            return self.allocator.num_free >= n_blocks

        while not ok():
            if not self._evict_one():
                return False
        return True

    def _enforce_budget(self) -> None:
        if self.max_cached_blocks is None:
            return
        while self.cached_blocks > self.max_cached_blocks:
            if not self._evict_one():
                # Every resident block is in use. Over budget, but evicting
                # would be R7. The budget is a target, never a licence.
                break

    def clear(self) -> int:
        """
        Drop every evictable node. Blocks still in use by a sequence stay.
        Returns how many were freed. Tests and shutdown.
        """
        freed = 0
        while self._evict_one():
            freed += 1
        return freed

    # -- copy-on-write ------------------------------------------------------

    def ensure_writable(self, seq: SequenceBlocks, block_index: int) -> int:
        """
        Make `seq`'s block at `block_index` privately writable. Returns the block
        id to write into (possibly a new one).

        WHEN THIS TRIGGERS: only on the LAST, PARTIALLY-FILLED block of a shared
        prefix. A full shared block is never written to — it is full — and this
        method refuses any other index precisely so that "full shared blocks are
        never written" is enforced rather than assumed. Under strict
        block-granularity reuse the prefill path never reaches here at all (see
        the module docstring); the guard is what keeps that true as the system
        grows partial-block reuse, forks, or parallel sampling.

        The sequence is untouched if it is the sole owner: a refcount of 1 means
        nobody else can observe the write.
        """
        if not 0 <= block_index < len(seq.block_ids):
            raise IndexError(
                f"block_index {block_index} outside sequence's {len(seq.block_ids)} blocks"
            )
        if block_index != len(seq.block_ids) - 1:
            raise ValueError(
                f"copy-on-write requested for block {block_index} of "
                f"{len(seq.block_ids)}, which is not the sequence's last block. Only the "
                "last, partially-filled block of a shared prefix is ever written to; a "
                "fully shared block is full by definition and a write into one is a bug "
                "in the caller, not a case to copy around."
            )

        old = seq.block_ids[block_index]
        if self.allocator.refcount(old) == 1:
            return old      # sole owner; nobody can observe the write

        # A fresh block, the shared prefix copied across, then the old reference
        # released. Order matters: allocate before freeing, or a failed
        # allocation leaves the sequence pointing at a block it no longer holds.
        if self._block_copy is None:
            raise RuntimeError(
                f"block {old} needs copy-on-write (refcount "
                f"{self.allocator.refcount(old)}) but no block_copy callable was "
                "supplied. Repointing without copying would give this sequence a block "
                "whose prefix KV was never written — fluent output, wrong attention."
            )
        new = self.allocator.allocate(1)[0]
        self._block_copy(old, new)
        self.allocator.free([old])
        seq.block_ids[block_index] = new
        self.stats.cow_copies += 1
        return new

    # -- reporting ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """
        Everything methodology §7 asks to be instrumented, in one dict.

        Note what is NOT here: a hit rate on its own. §7 is explicit that "a hit
        rate without the workload's sharing distribution attached is not a
        number", so the caller is expected to publish this next to the
        workload's realized sharing structure.
        """
        return {
            "enabled": self.enabled,
            "block_size": self.block_size,
            "lookups": self.stats.lookups,
            "requests_with_a_hit": self.stats.hits,
            "request_hit_rate": self.stats.request_hit_rate,
            "blocks_reused": self.stats.blocks_reused,
            "blocks_required": self.stats.blocks_required,
            "block_hit_rate": self.stats.hit_rate,
            "mean_partial_hit_depth": self.stats.mean_partial_hit_depth,
            "max_partial_hit_depth": max(self.stats.partial_hit_depths, default=0),
            "partial_block_truncations": self.stats.partial_block_truncations,
            "tokens_matched_but_recomputed": self.stats.truncated_tokens,
            "inserts": self.stats.inserts,
            "evictions": self.stats.evictions,
            "cow_copies": self.stats.cow_copies,
            "node_count": self.num_nodes,
            "max_node_depth": self.max_depth,
            "cached_blocks": self.cached_blocks,
            "shared_blocks_refcount_gt_1": self.shared_blocks,
            "definitions": {
                "block_hit_rate": "blocks_reused / blocks_required_at_prefill (§7)",
                "partial_hit_depth": "blocks matched before divergence, per request",
                "shared_blocks_refcount_gt_1": (
                    "cached blocks ALSO held by at least one live sequence right now"
                ),
            },
        }

    def check_invariants(self) -> None:
        """
        Structural check over the trie. O(nodes). Run it in tests and in debug
        benchmarks — R7's mitigation clause asks for exactly this.
        """
        for node in self._nodes:
            if node.parent is None:
                raise AssertionError(f"node at depth {node.depth} has no parent")
            if node.parent.children.get(node.key) is not node:
                raise AssertionError(
                    f"node at depth {node.depth} is not reachable from its parent"
                )
            if node.parent is not self.root and node.parent not in self._nodes:
                raise AssertionError("node's parent is not resident")
            if node.depth != node.parent.depth + 1:
                raise AssertionError("depth is not parent.depth + 1")
            if len(node.key) != self.block_size:
                raise AssertionError(
                    f"node key spans {len(node.key)} tokens, not one block "
                    f"({self.block_size}). An edge must be exactly one block."
                )
            rc = self.allocator.refcount(node.block_id)
            if rc < 1:
                raise AssertionError(
                    f"cached block {node.block_id} has refcount {rc}; the cache's own "
                    "reference is missing, so the block may already be handed out"
                )
            if self._by_block.get(node.block_id) is not node:
                raise AssertionError(f"block {node.block_id} is not mapped 1:1 to its node")
        if len(self._by_block) != len(self._nodes):
            raise AssertionError(
                f"{len(self._by_block)} blocks mapped but {len(self._nodes)} nodes resident"
            )


# ---------------------------------------------------------------------------
# Attaching a matched prefix to a sequence
# ---------------------------------------------------------------------------


def attach_prefix(seq: SequenceBlocks, block_ids: Sequence[int], n_tokens: int) -> None:
    """
    Install already-referenced cached blocks at the FRONT of a sequence's table.

    Why this reaches into `SequenceBlocks` fields instead of calling a method:
    `serving/memory/block_table.py` is frozen for this phase and has no "adopt
    blocks somebody else already increfed" API — `append()` always allocates.
    The write is confined to this one function, next to the reasoning, rather
    than being scattered through the scheduler.

    Contract, all of which the caller has already satisfied via `acquire()`:

      * every block in `block_ids` carries a reference now owned by `seq`, which
        `seq.free()` releases through the allocator like any other block;
      * `n_tokens == len(block_ids) * block_size` exactly — a whole number of
        blocks, because a partially reused block is the §9.1 bug;
      * the sequence is empty. Prefix reuse happens at admission, before any
        allocation, so a non-empty table means blocks are about to be orphaned.
    """
    if seq.is_freed:
        raise RuntimeError("cannot attach a cached prefix to a freed sequence")
    if seq.block_ids or seq.num_tokens:
        raise ValueError(
            f"sequence {seq.seq_id} already holds {len(seq.block_ids)} blocks / "
            f"{seq.num_tokens} tokens. A cached prefix is attached at admission, before "
            "anything is allocated; attaching later would strand the existing blocks."
        )
    if n_tokens != len(block_ids) * seq.block_size:
        raise ValueError(
            f"cached prefix claims {n_tokens} tokens over {len(block_ids)} blocks of "
            f"{seq.block_size}. Reuse is only ever a WHOLE number of blocks — a "
            "partially matching block must be recomputed (ARCHITECTURE.md §9.1)."
        )
    seq.block_ids.extend(block_ids)
    seq.num_tokens = n_tokens
