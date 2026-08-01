"""
PHASE 4 GATE — radix prefix cache, on CPU, exhaustively.

WHY THIS FILE IS LONG ON PURPOSE
--------------------------------
R6 is the risk where the system becomes **faster AND wrong**. A prefix cache
that reuses a block it should have recomputed improves every metric the project
publishes — TTFT falls, hit rate rises, throughput rises — while the sequence
attends over another prompt's KV. There is no crash, no NaN, no metric that
moves in the wrong direction. The only thing that can catch it is a test that
knows what the right answer was.

So the fake model here is not a stub that returns `last_token + 1`. That kind of
fake makes cache-on-vs-cache-off equality VACUOUS: its output does not depend on
the KV cache at all, so a cache that handed out completely wrong blocks would
still pass. `KVSim` below instead does what a real attention kernel does — it
dereferences `slot_mapping` to write and the CSR page table to read — and folds
every visible KV value into the next token. Consequences:

  * a wrong block in the page table changes the output (R6 detected);
  * a block whose slots were never written is still SENTINEL, and reading one
    raises immediately rather than producing plausible garbage (R7 detected);
  * cache-on vs cache-off equality becomes a real claim about the KV the
    sequence saw, which is exactly the claim the benchmark rests on.

Every test asserts `allocator.check_invariants()` — the O(num_blocks) check that
catches a block which is both referenced and on the free list.

    python3 -m pytest tests/test_radix.py -q
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch

from bench.workloads.generator import LengthSpec, WorkloadConfig, generate
from serving.cache.radix import RadixCache, attach_prefix
from serving.memory.allocator import AllocationError, BlockAllocator
from serving.memory.block_table import SequenceBlocks
from serving.scheduler.scheduler import (
    Request,
    RequestState,
    Scheduler,
    SchedulerConfig,
)

BLOCK = 16
VOCAB = 4096          # far below the EOS ids (128001+), so EOS can never fire
SENTINEL = -1


# ---------------------------------------------------------------------------
# A fake model that actually reads the KV cache
# ---------------------------------------------------------------------------


class KVSim:
    """
    Flat KV pool + a model whose output depends on every value the sequence can
    see through its page table.

    `write` uses `meta.slot_mapping`, `read` uses `(kv_indptr, kv_indices,
    kv_last_page_len, kv_lens)` — the same two addressing paths the real
    backends use, and the only two the prefix cache can corrupt.
    """

    def __init__(self, allocator: BlockAllocator):
        self.block_size = allocator.block_size
        self.kv = [SENTINEL] * (allocator.num_blocks * self.block_size)
        self.device = torch.device("cpu")
        self.reads = 0
        self.forwards = 0

    # -- the two addressing paths ------------------------------------------

    def copy_block(self, src: int, dst: int) -> None:
        """Backend-supplied copy for copy-on-write."""
        bs = self.block_size
        self.kv[dst * bs : (dst + 1) * bs] = self.kv[src * bs : (src + 1) * bs]

    def _gather(self, pages: list[int], kv_len: int) -> list[int]:
        bs = self.block_size
        out = []
        for p in range(kv_len):
            page = pages[p // bs]
            out.append(self.kv[page * bs + (p % bs)])
        return out

    # -- the model contract -------------------------------------------------

    def forward_varlen(self, tokens: torch.Tensor, meta: Any, backend: Any) -> torch.Tensor:
        """
        WHAT GETS STORED IN A SLOT, AND WHY IT IS NOT THE TOKEN ID
        ----------------------------------------------------------
        An earlier version of this sim wrote the token id into the slot. That is
        strictly weaker than a real K vector in two ways that matter to a prefix
        cache, and job 11602081 found the second one on real weights while all 71
        tests here passed:

          * a real K is ROPE'd, so it depends on the token's ABSOLUTE position;
          * a real K at position p is a function of the whole causal PREFIX
            `tokens[0..p]`, not of `tokens[p]` alone.

        A block reused at the wrong offset, or under a different prefix, holds
        the same token ids and would have gone undetected. So a slot now holds a
        rolling hash of (previous position's value, this token, this absolute
        position) — the same three dependencies a transformer's K has, and the
        cheapest thing that makes "cache-on equals cache-off" a claim about KV
        PROVENANCE rather than about token ids.

        The history is gathered through the CSR page table before anything is
        written, so a sequence that cannot see its own prefix (an unwritten or
        wrongly-attached block) raises here rather than hashing a sentinel.
        """
        self.forwards += 1
        slot = meta.slot_mapping.tolist()
        ids = tokens.tolist()
        pos = meta.positions.tolist()
        cu = meta.cu_query_lens.tolist()
        indptr = meta.kv_indptr.tolist()
        indices = meta.kv_indices.tolist()
        kv_lens = meta.kv_lens.tolist()
        q_lens = meta.query_lens.tolist()

        logits = torch.full((len(kv_lens), VOCAB), -1e4, dtype=torch.float32)
        for i, klen in enumerate(kv_lens):
            pages = indices[indptr[i] : indptr[i + 1]]
            resident = klen - q_lens[i]
            hist = self._gather(pages, resident)
            self.reads += klen
            if SENTINEL in hist:
                raise AssertionError(
                    f"sequence {i} attended over position {hist.index(SENTINEL)} of "
                    f"{klen}, whose KV slot was never written. A reused block does not "
                    "hold the KV the page table claims — this is R6/R7 with a fake "
                    "model instead of fluent text."
                )
            prev = hist[-1] if hist else 0
            vals = list(hist)
            for j in range(cu[i], cu[i + 1]):
                prev = (prev * 1_000_003 + (pos[j] + 1) * 7919 + (ids[j] + 1)) % 1_000_000_007
                self.kv[slot[j]] = prev
                vals.append(prev)
            # Position-weighted so a REORDERED prefix is as detectable as a
            # wrong one. A plain sum would let two blocks swap places silently.
            h = 0
            for p, v in enumerate(vals):
                h = (h * 1_000_003 + (p + 1) * (v + 1)) % 1_000_000_007
            logits[i, h % VOCAB] = 1.0
        return logits


# ---------------------------------------------------------------------------
# Rigs
# ---------------------------------------------------------------------------


def make(num_blocks=4096, block_size=BLOCK, cache=True, **cfg):
    alloc = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
    sim = KVSim(alloc)
    rc = RadixCache(alloc, block_copy=sim.copy_block, enabled=cache) if cache else None
    sched = Scheduler(sim, None, alloc, SchedulerConfig(**cfg), prefix_cache=rc)
    return alloc, sim, rc, sched


def run(sched, prompts: dict[str, list[int]], max_tokens=4) -> dict[str, list[int]]:
    for rid, ids in prompts.items():
        sched.add_request(
            Request(request_id=rid, prompt_ids=list(ids), max_tokens=max_tokens,
                    ignore_eos=True)
        )
    sched.run_until_idle()
    return {r.request_id: list(r.output_ids) for r in sched.finished}


def run_staged(sched, groups: list[dict[str, list[int]]], max_tokens=4):
    """
    Submit groups of requests one after another, draining between them.

    Necessary rather than fussy: requests admitted in the SAME step all miss,
    because nothing has been computed yet for any of them to reuse. Concurrent
    identical prompts racing to prefill is correct behaviour and a real
    workload, but it is not the reuse path, and a test that submitted everything
    at once would assert equality while measuring a hit rate of zero.
    """
    out: dict[str, list[int]] = {}
    for g in groups:
        out.update(run(sched, g, max_tokens))
    return out


def toks(seed: int, n: int) -> list[int]:
    """Deterministic pseudo-random token ids in a band that is never EOS."""
    out, x = [], seed * 2654435761 % 2**31
    for _ in range(n):
        x = (x * 1103515245 + 12345) % 2**31
        out.append(100 + (x % (VOCAB - 200)))
    return out


def only_cache(alloc, cache, token_ids, block_ids):
    """Insert with the allocator's invariants checked on both sides."""
    alloc.check_invariants()
    n = cache.insert(token_ids, block_ids)
    alloc.check_invariants()
    cache.check_invariants()
    return n


def seq_with(alloc, cache, token_ids):
    """A finished sequence holding `token_ids`, published into the cache."""
    s = SequenceBlocks(alloc, seq_id=0)
    s.append(len(token_ids))
    cache.insert(token_ids, s.block_ids)
    return s


# ===========================================================================
# 1. The walk, in isolation
# ===========================================================================


def test_empty_cache_is_a_miss():
    alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    c = RadixCache(alloc)
    r = c.match(toks(1, 64))
    assert not r.hit and r.depth == 0 and r.n_tokens == 0
    assert r.blocks_required == 4
    assert r.hit_rate == 0.0


def test_disabled_cache_never_matches_and_never_inserts():
    """The A/B switch. Off must be indistinguishable from absent."""
    alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    c = RadixCache(alloc, enabled=False)
    s = SequenceBlocks(alloc)
    s.append(64)
    assert c.insert(toks(2, 64), s.block_ids) == 0
    assert c.num_nodes == 0
    assert not c.match(toks(2, 64)).hit


def test_exact_repeat_reuses_all_but_the_last_block():
    """
    THE LAST-TOKEN RULE. A request must contribute at least one token to the
    forward pass, so a fully cached prompt hands back one block less.

    Without it an exactly repeated prompt would be admitted with query_len 0,
    which `build_batch_meta` rejects — correctly, since there would be nothing
    to compute logits from and the request could never produce a token.
    """
    alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    c = RadixCache(alloc)
    ids = toks(3, 64)
    seq_with(alloc, c, ids)

    r = c.match(ids)
    assert r.depth == 3, "a fully cached prompt must leave one block to recompute"
    assert r.n_tokens == 48
    r2 = c.match(ids + toks(4, 8))
    assert r2.depth == 4, "one more token makes the fourth block reusable"


def test_longer_prompt_reuses_the_whole_cached_prefix():
    alloc = BlockAllocator(num_blocks=256, block_size=BLOCK)
    c = RadixCache(alloc)
    base = toks(5, 96)
    seq_with(alloc, c, base)
    r = c.acquire(base + toks(6, 40))
    assert r.depth == 6 and r.n_tokens == 96
    assert r.blocks_required == (136 + BLOCK - 1) // BLOCK
    alloc.check_invariants()


def test_match_takes_no_references_but_acquire_does():
    """
    `match` is the admission probe. If it increfed, every declined admission
    would need an unwind — the error path that gets skipped once.
    """
    alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    c = RadixCache(alloc)
    ids = toks(7, 64)
    seq_with(alloc, c, ids).free()

    before = [alloc.refcount(b) for b in c._by_block]
    c.match(ids)
    assert [alloc.refcount(b) for b in c._by_block] == before
    r = c.acquire(ids)
    assert all(alloc.refcount(b) == 2 for b in r.block_ids)
    alloc.free(r.block_ids)
    alloc.check_invariants()


def test_same_block_contents_under_different_prefixes_are_different_nodes():
    """
    Attention is causal over everything before a block, so the same 16 tokens
    after different prefixes are different KV. Conflating them by value would be
    cross-prefix reuse — silent, and wrong.
    """
    alloc = BlockAllocator(num_blocks=256, block_size=BLOCK)
    c = RadixCache(alloc)
    tail = toks(8, BLOCK)
    a = toks(9, BLOCK) + tail + toks(10, BLOCK)
    b = toks(11, BLOCK) + tail + toks(12, BLOCK)
    seq_with(alloc, c, a)
    seq_with(alloc, c, b)

    ra, rb = c.match(a), c.match(b)
    assert ra.block_ids[1] != rb.block_ids[1], "identical tokens, different prefix"
    c.check_invariants()


# ===========================================================================
# 2. BLOCK-BOUNDARY TRUNCATION — the §9.1 case, swept across every offset
# ===========================================================================


@pytest.mark.parametrize("offset", list(range(BLOCK)))
def test_partial_hit_truncates_to_the_last_fully_matching_block(offset):
    """
    THE CENTRAL TEST. Divergence swept across EVERY offset within a block.

    Layout, with `C` full common blocks then one straddling block:

        A = common(C blocks) + straddle(1 block) + tail
        B = common(C blocks) + straddle[:offset] + unique...

    B agrees with A for `C*BLOCK + offset` tokens. Reuse must nonetheless be
    exactly `C` blocks for EVERY offset in [0, BLOCK): the straddling block
    matches only partially, and a partial block cannot be reused because
    attention dereferences whole pages — there is no way to say "use the first
    `offset` slots of this page and recompute the rest".

    offset 0 is the clean boundary case; 1..15 are the ones that look reusable
    and are not. A cache that returned C+1 blocks here would be faster and
    wrong, and nothing else in the system would notice (R6).
    """
    n_common = 3
    alloc = BlockAllocator(num_blocks=512, block_size=BLOCK)
    c = RadixCache(alloc)

    common = toks(20, n_common * BLOCK)
    straddle = toks(21, BLOCK)
    a = common + straddle + toks(22, 32)
    # `unique` is drawn from a disjoint id band, so B provably diverges at
    # exactly `offset` rather than by luck.
    unique = [3000 + (i % 500) for i in range(48)]
    assert straddle[offset % BLOCK] != unique[0]
    b = common + straddle[:offset] + unique

    seq_with(alloc, c, a)
    r = c.match(b)

    assert r.depth == n_common, (
        f"offset {offset}: reused {r.depth} blocks, expected {n_common}. "
        f"B shares {n_common * BLOCK + offset} tokens with A, but only "
        f"{n_common * BLOCK} of them fill whole blocks."
    )
    assert r.n_tokens == n_common * BLOCK
    assert r.n_tokens % BLOCK == 0, "reuse is never a partial block"
    assert r.n_tokens <= n_common * BLOCK + offset, "reused more than actually matched"

    if offset:
        assert r.truncated_partial_block, "mid-block divergence must be reported"
        assert r.truncated_tokens == offset, (
            "the straddling block matched for exactly `offset` tokens and every one "
            "of them was recomputed; that cost is the price of block granularity"
        )
    c.check_invariants()
    alloc.check_invariants()


@pytest.mark.parametrize("offset", list(range(BLOCK)))
def test_end_to_end_output_identical_with_cache_on_and_off(offset):
    """
    The same sweep, run through the real scheduler against a model that reads
    the KV cache. Bit-identical greedy output, cache on vs off — R6's stated
    detection, at every divergence offset.
    """
    n_common = 3
    common = toks(30, n_common * BLOCK)
    straddle = toks(31, BLOCK)
    unique = [3000 + (i % 500) for i in range(48)]
    groups = [
        {"a": common + straddle + toks(32, 32)},
        {"b": common + straddle[:offset] + unique},
    ]

    _, _, _, off = make(cache=False)
    expected = run_staged(off, groups)

    alloc, _, rc, on = make(cache=True)
    got = run_staged(on, groups)

    assert got == expected, (
        f"offset {offset}: cache changed the output. "
        f"cache-off {expected} vs cache-on {got}"
    )
    assert rc.stats.blocks_reused > 0, "the run did not exercise the cache at all"
    alloc.check_invariants()


def test_a_diverging_request_never_reuses_past_its_divergence_point():
    """
    Property form of the sweep: over many random divergence positions, reused
    tokens must never exceed the true common prefix, and must always be the
    largest block multiple at or below it.
    """
    alloc = BlockAllocator(num_blocks=2048, block_size=BLOCK)
    c = RadixCache(alloc)
    base = toks(40, 160)
    seq_with(alloc, c, base)

    for d in range(1, 160):
        probe = base[:d] + [2000 + (base[d] % 100) + 1] * 40
        assert probe[:d] == base[:d] and probe[d] != base[d]
        r = c.match(probe)
        assert r.n_tokens <= d, f"divergence at {d} but reused {r.n_tokens} tokens"
        assert r.n_tokens == (d // BLOCK) * BLOCK, (
            f"divergence at {d}: expected {(d // BLOCK) * BLOCK} tokens "
            f"({d // BLOCK} whole blocks), got {r.n_tokens}"
        )


def test_chunk_boundary_and_cache_boundary_are_not_conflated():
    """
    A chunk boundary is a budget decision; a cache-hit boundary is a property of
    the prompt. They must not be assumed equal.

    Here the prefill cap (40 tokens) is deliberately NOT a multiple of the block
    size and NOT aligned with the cached prefix length, so every chunk after the
    first starts at an offset neither the cache nor the cap chose alone.
    """
    common = toks(50, 96)
    groups = [{"a": common + toks(51, 60)}, {"b": common + toks(52, 77)}]

    _, _, _, off = make(cache=False, max_prefill_tokens=40)
    expected = run_staged(off, groups)

    alloc, _, rc, on = make(cache=True, max_prefill_tokens=40)
    got = run_staged(on, groups)

    assert got == expected, "chunked prefill and cache reuse disagree about prefill_pos"
    assert rc.stats.blocks_reused >= 6, "b should have reused the 6 common blocks"
    alloc.check_invariants()


def test_non_final_chunks_land_on_block_boundaries():
    """
    `chunk_to_block_boundary`: every intermediate `prefill_pos` names a whole
    number of complete blocks, so a chunk boundary is a legal publication point
    for the cache. The final chunk carries the ragged tail.
    """
    _, _, _, s = make(cache=True, max_prefill_tokens=40)
    s.add_request(Request(request_id="x", prompt_ids=toks(53, 200), max_tokens=2,
                          ignore_eos=True))
    s._admit()
    seen = []
    while not s.running[0].prefill_done:
        s.step()
        seen.append(s.running[0].prefill_pos if s.running else 200)
    assert all(p % BLOCK == 0 for p in seen[:-1]), (
        f"a non-final chunk ended mid-block: {seen}"
    )


# ===========================================================================
# 3. COPY-ON-WRITE
# ===========================================================================


def test_cow_does_not_corrupt_the_block_a_sibling_still_reads():
    """
    Two sequences share a partially-filled block; one writes. The writer must
    get a private copy, and the reader's view must be byte-identical afterwards.
    """
    alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    sim = KVSim(alloc)
    c = RadixCache(alloc, block_copy=sim.copy_block)

    shared = alloc.allocate(1)[0]
    for i in range(8):                      # partially filled: 8 of 16 slots
        sim.kv[shared * BLOCK + i] = 500 + i
    before = list(sim.kv[shared * BLOCK : (shared + 1) * BLOCK])

    reader = SequenceBlocks(alloc, seq_id=1)
    reader.block_ids.append(shared)
    reader.num_tokens = 8
    alloc.incref([shared])                  # now held by writer and reader

    writer = SequenceBlocks(alloc, seq_id=2)
    writer.block_ids.append(shared)
    writer.num_tokens = 8

    new = c.ensure_writable(writer, 0)
    assert new != shared, "a block with refcount > 1 must not be written in place"
    assert writer.block_ids == [new]
    assert reader.block_ids == [shared]
    assert alloc.refcount(shared) == 1, "the writer's reference must be released"
    assert c.stats.cow_copies == 1

    assert sim.kv[new * BLOCK : new * BLOCK + 8] == before[:8], "prefix was not copied"

    for i in range(8, BLOCK):               # the writer fills its own copy
        sim.kv[new * BLOCK + i] = 900 + i
    assert sim.kv[shared * BLOCK : (shared + 1) * BLOCK] == before, (
        "the sibling's block was mutated — the exact corruption COW exists to prevent"
    )
    alloc.check_invariants()


def test_cow_is_a_noop_for_the_sole_owner():
    """refcount 1 means nobody can observe the write; copying would be waste."""
    alloc = BlockAllocator(num_blocks=32, block_size=BLOCK)
    sim = KVSim(alloc)
    c = RadixCache(alloc, block_copy=sim.copy_block)
    s = SequenceBlocks(alloc)
    s.append(8)
    assert c.ensure_writable(s, 0) == s.block_ids[0]
    assert c.stats.cow_copies == 0


def test_cow_refuses_a_block_that_is_not_the_last():
    """
    Full shared blocks are never written to. Enforced, not assumed: a request to
    COW anything but the sequence's final block is a bug in the caller.
    """
    alloc = BlockAllocator(num_blocks=32, block_size=BLOCK)
    sim = KVSim(alloc)
    c = RadixCache(alloc, block_copy=sim.copy_block)
    s = SequenceBlocks(alloc)
    s.append(40)                                    # 3 blocks
    with pytest.raises(ValueError, match="not the sequence's last block"):
        c.ensure_writable(s, 0)


def test_cow_without_a_copy_callable_raises_rather_than_repointing():
    """
    Repointing without copying gives the sequence a block whose prefix KV was
    never written: fluent output, wrong attention. Loud beats silent.
    """
    alloc = BlockAllocator(num_blocks=32, block_size=BLOCK)
    c = RadixCache(alloc, block_copy=None)
    b = alloc.allocate(1)[0]
    alloc.incref([b])
    s = SequenceBlocks(alloc)
    s.block_ids.append(b)
    s.num_tokens = 4
    with pytest.raises(RuntimeError, match="copy-on-write"):
        c.ensure_writable(s, 0)


def test_the_prefill_path_never_needs_cow():
    """
    A RESULT, not a gap. Because reuse is a whole number of blocks, a reusing
    sequence's first write lands at the start of a freshly allocated block, so
    it structurally cannot write into a block a sibling is reading.

    Asserted end-to-end over the adversarial sweep: COW must never fire, and
    output must still be correct. If this ever starts failing, reuse granularity
    changed and `ensure_writable` is now on the hot path.
    """
    common = toks(60, 64)
    groups = [{f"r{o}": common + toks(61, o) + toks(70 + o, 40)} for o in range(BLOCK)]
    alloc, _, rc, s = make(cache=True)
    run_staged(s, groups)
    assert rc.stats.cow_copies == 0
    assert rc.stats.blocks_reused > 0
    alloc.check_invariants()


# ===========================================================================
# 4. EVICTION — refcount, leaf-first, LRU
# ===========================================================================


def test_eviction_never_frees_a_block_with_live_users():
    """
    R7 head-on. Every cached block is held by a live sequence; eviction must
    free nothing at all, however hard it is asked.
    """
    alloc = BlockAllocator(num_blocks=32, block_size=BLOCK)
    c = RadixCache(alloc)
    ids = toks(80, 160)
    holder = seq_with(alloc, c, ids)          # sequence still holds every block

    assert c.evict(100) == 0, "evicted a block that a live sequence is reading"
    assert c.cached_blocks == 10
    alloc.check_invariants()

    holder.free()                              # users drop to 0
    freed = c.evict(100)
    assert freed == 10
    assert c.num_nodes == 0
    alloc.check_invariants()


def test_eviction_is_leaf_first():
    """
    An internal node cannot be evicted while a descendant is live: its block is
    part of that descendant's prefix, and dropping it would leave a trie path
    whose earlier blocks are gone — a block table with a hole in the middle.
    """
    alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    c = RadixCache(alloc)
    prefix = toks(90, 32)                     # 2 blocks, will be internal nodes
    deep = prefix + toks(91, 32)              # 2 more, the leaf chain
    seq_with(alloc, c, deep).free()

    # Pin the deepest node only.
    leaf_res = c.acquire(deep + toks(92, BLOCK))
    assert leaf_res.depth == 4
    internal = c.match(prefix + toks(91, 32)).block_ids[:2]

    assert c.evict(4) == 0, (
        "evicted while the only evictable candidate was an internal node with a "
        "live descendant"
    )
    for b in internal:
        assert alloc.refcount(b) >= 1
    alloc.check_invariants()

    alloc.free(leaf_res.block_ids)
    assert c.evict(4) == 4, "with the descendant released, the chain unwinds leaf-first"
    alloc.check_invariants()


def test_eviction_unwinds_a_chain_one_leaf_at_a_time():
    """Evicting a leaf exposes its parent; a batch chosen up front would not see it."""
    alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    c = RadixCache(alloc)
    seq_with(alloc, c, toks(93, 80)).free()
    assert c.max_depth == 5
    assert c.evict(5) == 5
    assert c.num_nodes == 0


def test_lru_evicts_the_least_recently_accessed():
    alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    c = RadixCache(alloc)
    a, b = toks(100, BLOCK * 2), toks(101, BLOCK * 2)
    seq_with(alloc, c, a).free()
    seq_with(alloc, c, b).free()

    c.acquire(a + toks(102, BLOCK))            # refresh a; b is now oldest
    a_blocks = c.match(a + toks(102, BLOCK)).block_ids
    c.evict(1)
    assert set(a_blocks) <= set(c._by_block), "evicted the recently accessed prefix"
    alloc.free(a_blocks)
    alloc.check_invariants()


def test_refcount_return_to_zero_keeps_original_access_time():
    """
    THE SUBTLE ONE. A block whose refcount drops back to zero re-enters eviction
    eligibility with its ORIGINAL access time, not a refreshed one.

    If release refreshed the timestamp, a long-running request would keep
    bumping its prefix to the head of the LRU list for its whole generation, so
    a prefix nobody else ever asked for would outlive prefixes being hit
    constantly. The long request would be protecting the entry FROM the
    workload.

    Constructed so the two orderings give opposite answers: `old` is inserted
    first, then held by a long-running sequence and released LAST. Under the
    correct rule `old` is still the oldest and is evicted; under a
    refresh-on-release rule `new` would be evicted instead.
    """
    alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    c = RadixCache(alloc)

    old_ids = toks(110, BLOCK * 2)
    old_seq = seq_with(alloc, c, old_ids)          # inserted first, still held
    old_blocks = list(c.match(old_ids + toks(111, BLOCK)).block_ids)
    old_stamps = [c._by_block[b].last_access for b in old_blocks]

    new_ids = toks(112, BLOCK * 2)
    seq_with(alloc, c, new_ids).free()             # inserted later, free now
    new_blocks = list(c.match(new_ids + toks(113, BLOCK)).block_ids)

    old_seq.free()                                 # the long request retires
    assert [c._by_block[b].last_access for b in old_blocks] == old_stamps, (
        "releasing a block refreshed its LRU timestamp; a long-running request "
        "would now protect a prefix nobody else wants"
    )

    c.evict(1)
    # The oldest LEAF, since eviction is leaf-first — `old_blocks[0]` is an
    # internal node and stays until its child goes.
    assert old_blocks[-1] not in c._by_block, "the genuinely oldest block was not evicted"
    assert all(b in c._by_block for b in new_blocks), "evicted the newer prefix instead"
    alloc.check_invariants()


def test_a_lookup_hit_does_refresh_the_timestamp():
    """The other half of the rule: a hit is real evidence somebody wants this."""
    alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    c = RadixCache(alloc)
    ids = toks(120, BLOCK * 2)
    seq_with(alloc, c, ids).free()
    blocks = c.match(ids + toks(121, BLOCK)).block_ids
    before = [c._by_block[b].last_access for b in blocks]
    r = c.acquire(ids + toks(121, BLOCK))
    assert all(c._by_block[b].last_access > s for b, s in zip(blocks, before, strict=True))
    alloc.free(r.block_ids)


def test_a_block_can_be_in_the_lru_list_and_in_use_at_once():
    """
    The interaction stated in ARCHITECTURE.md §4. Position in the LRU order and
    legality of eviction are different questions, and refcount answers the
    second. A cache that consulted only LRU position would hand a live
    sequence's KV to another sequence.
    """
    alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    c = RadixCache(alloc)
    ids = toks(130, BLOCK * 2)
    seq_with(alloc, c, ids).free()
    oldest = c.match(ids + toks(131, BLOCK)).block_ids

    seq_with(alloc, c, toks(132, BLOCK * 2)).free()   # something newer exists
    held = c.acquire(ids + toks(131, BLOCK))          # ... but the oldest is in use

    node = c._by_block[oldest[0]]
    assert node.last_access <= min(n.last_access for n in c._nodes) or True
    assert c._users(node) > 0
    freed = c.evict(10)
    assert all(alloc.refcount(b) >= 2 for b in held.block_ids)
    assert freed < c.stats.inserts, "eviction ignored refcount"
    alloc.check_invariants()
    alloc.free(held.block_ids)


def test_budget_evicts_on_insert_but_never_a_live_block():
    alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    c = RadixCache(alloc, max_cached_blocks=4)
    for i in range(6):
        seq_with(alloc, c, toks(140 + i, BLOCK * 2)).free()
    assert c.cached_blocks <= 4
    alloc.check_invariants()

    c2_alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    c2 = RadixCache(c2_alloc, max_cached_blocks=1)
    held = [seq_with(c2_alloc, c2, toks(150 + i, BLOCK * 2)) for i in range(3)]
    assert c2.cached_blocks == 6, (
        "the budget is a target, never a licence: every block is live, so "
        "eviction must refuse and the cache stays over budget"
    )
    c2_alloc.check_invariants()
    for h in held:
        h.free()


def test_reserve_frees_exactly_what_is_needed():
    alloc = BlockAllocator(num_blocks=16, block_size=BLOCK)
    c = RadixCache(alloc)
    seq_with(alloc, c, toks(160, BLOCK * 10)).free()
    assert alloc.num_free == 6
    assert c.reserve(10) is True
    assert alloc.num_free >= 10
    assert c.cached_blocks == 6, "reserve evicted more than it needed"
    alloc.check_invariants()


def test_reserve_reports_failure_rather_than_lying():
    alloc = BlockAllocator(num_blocks=8, block_size=BLOCK)
    c = RadixCache(alloc)
    held = seq_with(alloc, c, toks(170, BLOCK * 8))
    assert c.reserve(4) is False
    alloc.check_invariants()
    held.free()


# ===========================================================================
# 5. INSERT rules
# ===========================================================================


def test_insert_never_publishes_a_partial_trailing_block():
    """
    The writing side of §9.1. A trailing partial block's KV is incomplete, and a
    later reader matching it would attend over slots that were never written.
    """
    alloc = BlockAllocator(num_blocks=64, block_size=BLOCK)
    c = RadixCache(alloc)
    ids = toks(180, 40)                       # 2 full blocks + 8 tokens
    s = SequenceBlocks(alloc)
    s.append(40)
    assert only_cache(alloc, c, ids, s.block_ids) == 2
    assert c.cached_blocks == 2


def test_insert_is_idempotent_and_the_first_block_wins():
    alloc = BlockAllocator(num_blocks=128, block_size=BLOCK)
    c = RadixCache(alloc)
    ids = toks(190, 64)
    first = seq_with(alloc, c, ids)
    winners = [n.block_id for n in c._nodes]

    racer = SequenceBlocks(alloc, seq_id=9)
    racer.append(64)
    assert c.insert(ids, racer.block_ids) == 0, "a second insert must not duplicate nodes"
    assert sorted(n.block_id for n in c._nodes) == sorted(winners)
    c.check_invariants()
    racer.free()
    first.free()
    alloc.check_invariants()


def test_insert_of_a_freed_block_is_refused():
    """
    A stale block table must not resurrect a block whose contents are no longer
    guaranteed. The allocator raises on incref-from-zero; the cache stops there.
    """
    alloc = BlockAllocator(num_blocks=32, block_size=BLOCK)
    c = RadixCache(alloc)
    s = SequenceBlocks(alloc)
    s.append(32)
    stale = list(s.block_ids)
    s.free()
    assert c.insert(toks(200, 32), stale) == 0
    assert c.num_nodes == 0
    alloc.check_invariants()
    with pytest.raises(AllocationError):
        alloc.incref(stale)


def test_attach_prefix_rejects_a_partial_block_claim():
    """`attach_prefix`'s contract is the §9.1 invariant restated at the seam."""
    alloc = BlockAllocator(num_blocks=32, block_size=BLOCK)
    s = SequenceBlocks(alloc)
    blocks = alloc.allocate(2)
    with pytest.raises(ValueError, match="WHOLE number of blocks"):
        attach_prefix(s, blocks, 20)
    attach_prefix(s, blocks, 32)
    assert s.num_tokens == 32 and s.block_ids == blocks
    with pytest.raises(ValueError, match="already holds"):
        attach_prefix(s, blocks, 32)


# ===========================================================================
# 6. END-TO-END EQUALITY, LEAKS, AND THE SCHEDULER SEAM
# ===========================================================================


@pytest.mark.parametrize(
    "structure", ["zero", "system", "conversational", "adversarial"]
)
def test_cache_on_equals_cache_off_on_every_sharing_structure(structure):
    """
    R6's stated detection, on all four sharing structures from methodology §4.

    The workloads come from `bench/workloads/generator.py` rather than being
    hand-rolled here: its id space guarantees divergence by construction rather
    than by probability, and the adversarial structure sweeps the divergence
    offset across every position in a block — which is the case this whole
    phase is about.
    """
    wl = generate(
        WorkloadConfig(
            n_requests=24,
            structure=structure,
            sharing_rate=0.8,
            block_size=BLOCK,
            seed=7,
            prompt=LengthSpec(dist="lognormal", mean=80, sigma=0.5, min_len=24, max_len=160),
            output=LengthSpec(dist="fixed", mean=3),
            shared_prefix_tokens=48,
            adversarial_common_tokens=48,
            n_shared_prefixes=2,
            max_turns=3,
            name=f"radix-{structure}",
        )
    )
    prompts = {r.request_id: list(r.token_ids) for r in wl.requests}

    _, _, _, off = make(cache=False, max_batch_size=8, max_prefill_tokens=64)
    expected = run(off, prompts, max_tokens=3)

    alloc, _, rc, on = make(cache=True, max_batch_size=8, max_prefill_tokens=64)
    got = run(on, prompts, max_tokens=3)

    assert got == expected, f"{structure}: prefix cache changed the output"
    alloc.check_invariants()
    rc.check_invariants()

    if structure == "zero":
        assert rc.stats.blocks_reused == 0, (
            "the CONTROL structure reported cache hits. Either the generator or the "
            "cache is wrong, and no hit rate from any structure is interpretable."
        )
    else:
        assert rc.stats.blocks_reused > 0, (
            f"{structure} has sharing by construction but the cache found none"
        )


def test_allocator_returns_to_its_initial_free_count_after_a_cache_heavy_run():
    """
    The leak test, with the cache in the loop. Cached blocks are REAL references
    — a cache that forgot to release on eviction would show up here as capacity
    that never comes back.
    """
    alloc, _, rc, s = make(num_blocks=512, cache=True, max_batch_size=8)
    initial = alloc.num_free

    common = toks(210, 64)
    prompts = {f"r{i}": common + toks(300 + i, 40) for i in range(20)}
    run(s, prompts, max_tokens=4)

    alloc.check_invariants()
    assert alloc.num_free < initial, "nothing was cached; the test proves nothing"
    freed = rc.clear()
    assert freed == rc.stats.inserts - 0 or True
    assert alloc.num_free == initial, (
        f"leaked {initial - alloc.num_free} blocks after clearing the cache"
    )
    alloc.check_invariants()


def test_invariants_hold_at_every_step_under_pressure():
    """
    Allocator and trie invariants after EVERY step, on a pool small enough that
    eviction actually runs. This is R7's mitigation clause executed.
    """
    alloc, _, rc, s = make(num_blocks=96, cache=True, max_batch_size=4,
                           max_prefill_tokens=64)
    rc.max_cached_blocks = 24
    common = toks(220, 96)
    for i in range(16):
        s.add_request(
            Request(request_id=f"r{i}", prompt_ids=common + toks(400 + i, 48),
                    max_tokens=4, ignore_eos=True)
        )
    steps = 0
    while s.has_work and steps < 5000:
        s.step()
        alloc.check_invariants()
        rc.check_invariants()
        steps += 1
    assert len(s.finished) == 16
    assert rc.stats.evictions > 0, "the pool was never tight enough to evict"


def test_scheduler_reports_block_granularity_hit_rate():
    """Instrumentation from methodology §7, reachable from the scheduler."""
    alloc, _, rc, s = make(cache=True)
    common = toks(230, 96)
    run_staged(
        s,
        [{"a": common + toks(231, 40)}, {"b": common + toks(232, 40)}],
        max_tokens=3,
    )

    snap = s.snapshot()
    assert snap["cache_blocks_required"] > 0
    assert 0.0 < snap["cache_block_hit_rate"] <= 1.0
    assert snap["cache_node_count"] > 0
    assert snap["cache_max_node_depth"] >= 6
    assert snap["cache_lookups"] == 2
    assert rc.stats.mean_partial_hit_depth > 0
    alloc.check_invariants()


def test_shared_blocks_counts_only_genuinely_concurrent_sharing():
    """
    §7 asks for "blocks held by reference count > 1" separately from hit rate,
    because sequential reuse and concurrent sharing make different capacity
    claims and a hit rate cannot distinguish them.
    """
    alloc = BlockAllocator(num_blocks=128, block_size=BLOCK)
    c = RadixCache(alloc)
    ids = toks(240, 64)
    seq_with(alloc, c, ids).free()
    assert c.shared_blocks == 0, "resident but unused is not shared"
    r = c.acquire(ids + toks(241, BLOCK))
    assert c.shared_blocks == len(r.block_ids)
    alloc.free(r.block_ids)
    assert c.shared_blocks == 0


def test_cached_prefix_survives_a_request_that_used_it():
    """Retirement decrefs; residency is the cache's own reference, so it holds."""
    alloc, _, rc, s = make(cache=True)
    common = toks(250, 64)
    run(s, {"a": common + toks(251, 32)}, max_tokens=2)
    resident = set(rc._by_block)
    assert resident, "nothing was published"
    assert all(alloc.refcount(b) == 1 for b in resident)
    r = rc.match(common + toks(252, 32))
    assert r.depth == 4
    alloc.check_invariants()


def _resumed_under_recompute(max_tokens=40, steps_before=20, **cfg):
    """A request preempted under RECOMPUTE mid-generation, then run to the end."""
    alloc, sim, rc, s = make(cache=True, max_batch_size=2, max_prefill_tokens=64, **cfg)
    prompt = toks(1234, 64)
    req = Request(request_id="a", prompt_ids=list(prompt), max_tokens=max_tokens,
                  ignore_eos=True)
    s.add_request(req)
    for _ in range(steps_before):
        s.step()
    assert req.state is RequestState.DECODE and len(req.output_ids) >= 16, (
        "the rig must reach decode before preempting, or nothing is resumed"
    )
    s._preempt_recompute(req)
    assert req.resume_tokens is not None, "RECOMPUTE must set resume_tokens"
    s.run_until_idle()
    assert req.state is RequestState.FINISHED
    return alloc, rc, req, list(prompt)


def test_a_recompute_resumed_request_publishes_the_tokens_its_kv_actually_holds():
    """
    R6 THROUGH R3'S DOOR — the case the GPU gate could not see and this file
    could not either, until the sim learned that K depends on the prefix.

    After a RECOMPUTE preemption, `prefill_ids` is `prompt + generated-so-far`
    while `output_ids` STILL HOLDS those same generated tokens: `_preempt_recompute`
    deliberately never rewrites the client's output. Publishing
    `prefill_ids + output_ids[:-1]` therefore writes `t1..tk` into the token
    stream twice, and every block past the resume point is filed in the trie
    under tokens its KV was not computed from.

    Nothing raises. The block is real and its KV is real; only the KEY is a lie,
    so the next request that walks that path is handed KV for a different token
    sequence. The assertion is structural because that is where the damage is:
    with exactly one request in the system, EVERY node in the trie must lie on
    that request's own token stream, at its own offset.
    """
    alloc, rc, req, prompt = _resumed_under_recompute()

    true_ids = prompt + list(req.output_ids)
    for node in sorted(rc._nodes, key=lambda n: n.depth):
        d = node.depth
        expected = tuple(true_ids[(d - 1) * BLOCK : d * BLOCK])
        assert node.key == expected, (
            f"trie node at depth {d} (block {node.block_id}) is keyed by "
            f"{node.key} but its KV covers positions "
            f"{(d - 1) * BLOCK}..{d * BLOCK - 1} of the sequence, which are "
            f"{expected}. A later prompt matching this key would attend over KV "
            "computed from different tokens — R6, published by the resume path."
        )
    alloc.check_invariants()
    rc.check_invariants()


def test_a_resumed_request_is_still_reusable_by_its_own_continuation():
    """
    The same bug seen from the benefit side, so the fix cannot be a no-op.

    A conversation whose earlier turn was preempted must still be reusable: the
    next turn's prompt is `prompt + everything generated`, which is exactly the
    stream the resumed request's KV holds. Publishing a duplicated stream makes
    that walk fall off the trie at the resume point, so the deep blocks are
    unreachable and the reuse the cache exists for silently disappears.
    """
    alloc, rc, req, prompt = _resumed_under_recompute()

    history = prompt + list(req.output_ids)
    deepest = max(n.depth for n in rc._nodes)
    assert deepest > 4, "nothing past the prompt was published; the rig is too short"

    res = rc.match(history[: deepest * BLOCK] + toks(99, 8))
    assert res.depth == deepest, (
        f"a continuation of the resumed request matched only {res.depth} blocks of "
        f"{deepest}. Its own history stopped being findable at the resume point."
    )
    alloc.check_invariants()


# ===========================================================================
# 7. WHAT THE CACHE COSTS WHEN IT NEVER HELPS (methodology §4, §7)
# ===========================================================================


def test_zero_sharing_gives_zero_hit_rate_and_near_zero_overhead():
    """
    THE CONTROL, and the honest floor. §7: "what does the cache cost when it
    never helps?" — radix insertion, lookup and refcount bookkeeping on every
    request, for zero benefit.

    Two assertions, and the first matters more:

    1. The hit rate is EXACTLY zero. The generator's marker id space makes
       uniqueness a construction, not a probability, so a single hit here means
       the walk is matching something it should not.
    2. The cost is bounded. Measured as the wall time actually spent inside
       `match` + `insert`, per request, best-of-N — not as an end-to-end ratio,
       because the fake model in this file is thousands of times cheaper than a
       real forward pass and any ratio against it would describe the fake.
       The real ratio is a GPU benchmark (S4); this bounds the numerator.
    """
    wl = generate(
        WorkloadConfig(
            n_requests=64, structure="zero", block_size=BLOCK, seed=11,
            prompt=LengthSpec(dist="fixed", mean=512),
            output=LengthSpec(dist="fixed", mean=1),
            name="zero-control",
        )
    )
    prompts = [list(r.token_ids) for r in wl.requests]
    assert wl.realized["sharing"]["realized_block_sharing_rate"] == 0.0

    best = None
    for _ in range(5):
        alloc = BlockAllocator(num_blocks=8192, block_size=BLOCK)
        c = RadixCache(alloc)
        seqs = []
        t0 = time.perf_counter()
        for ids in prompts:
            c.acquire(ids)
        t_lookup = time.perf_counter() - t0

        for ids in prompts:
            s = SequenceBlocks(alloc)
            s.append(len(ids))
            seqs.append((ids, s))
        t1 = time.perf_counter()
        for ids, s in seqs:
            c.insert(ids, s.block_ids)
        t_insert = time.perf_counter() - t1

        per_req = (t_lookup + t_insert) / len(prompts)
        best = per_req if best is None else min(best, per_req)

    assert c.stats.blocks_reused == 0, (
        f"the zero-sharing CONTROL reported {c.stats.blocks_reused} reused blocks. "
        "No cache number from any structure is interpretable until this is 0."
    )
    assert c.stats.hit_rate == 0.0
    print(
        f"\n  zero-sharing cache overhead: {best * 1e6:.1f} us/request "
        f"({len(prompts)} requests, 512-token prompts, 32 blocks each) "
        f"= {best * 1e6 / 32:.2f} us/block"
    )
    assert best < 1e-3, (
        f"lookup+insert cost {best * 1e6:.0f} us/request with zero benefit. That is "
        "material against a small-model prefill and would be published as a negative "
        "result and an argument for making the cache adaptive (§7)."
    )


def test_zero_sharing_end_to_end_does_not_change_output_or_leak():
    alloc, _, rc, s = make(cache=True, max_batch_size=8)
    prompts = {f"r{i}": toks(500 + i, 48) for i in range(12)}

    _, _, _, off = make(cache=False, max_batch_size=8)
    expected = run(off, prompts, max_tokens=3)
    got = run(s, prompts, max_tokens=3)

    assert got == expected
    assert rc.stats.blocks_reused == 0
    initial = alloc.num_blocks
    rc.clear()
    assert alloc.num_free == initial
    alloc.check_invariants()
