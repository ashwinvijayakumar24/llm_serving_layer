"""
copy_block: the seam that makes copy-on-write real.

The radix cache owns block INDICES; a backend owns the TENSORS. When a sequence
diverges inside a block it shares, the cache allocates a fresh block and asks
the backend to duplicate the KV before anything writes into it.

Two failure modes, both silent:
  * no copy at all — the sequence writes into a block its sibling still reads
  * a LAYER-0-ONLY copy — layers 1..N-1 still point at the sibling's KV, so
    attention reads a mixture of two histories and produces fluent, wrong text

The second is the one a careless implementation actually ships, because layer 0
is what you check by hand. These tests read every layer.
"""

import pytest

torch = pytest.importorskip("torch")

from serving.backends.paged_torch import PagedTorchBackend  # noqa: E402


def backend(num_layers=4, num_blocks=8, block_size=4, n_kv=2, n_heads=4, head_dim=8):
    return PagedTorchBackend(
        num_layers=num_layers, num_blocks=num_blocks, block_size=block_size,
        n_kv_heads=n_kv, n_heads=n_heads, head_dim=head_dim,
        device="cpu", dtype=torch.float32,
    )


def fill(bk, block, base):
    """Give every layer a DIFFERENT pattern, so a layer-0-only copy is visible."""
    for layer in range(bk.num_layers):
        bk.k_pool[layer][block] = base + layer * 100
        bk.v_pool[layer][block] = base + layer * 100 + 50


def test_copies_every_layer():
    """THE TEST. A layer-0-only copy passes a naive check and fails this one."""
    bk = backend()
    fill(bk, 1, 7.0)
    fill(bk, 2, -1.0)
    bk.copy_block(1, 2)
    for layer in range(bk.num_layers):
        assert torch.equal(bk.k_pool[layer][2], bk.k_pool[layer][1]), f"K layer {layer} not copied"
        assert torch.equal(bk.v_pool[layer][2], bk.v_pool[layer][1]), f"V layer {layer} not copied"


def test_copy_is_a_copy_not_an_alias():
    """Writing to the destination must not disturb the source."""
    bk = backend()
    fill(bk, 0, 3.0)
    bk.copy_block(0, 5)
    before = bk.k_pool[2][0].clone()
    bk.k_pool[2][5] += 99.0
    assert torch.equal(bk.k_pool[2][0], before), "destination aliases the source"


def test_other_blocks_untouched():
    bk = backend()
    for b in range(bk.num_blocks):
        fill(bk, b, float(b))
    keep = {b: bk.k_pool[1][b].clone() for b in range(bk.num_blocks) if b not in (1, 2)}
    bk.copy_block(1, 2)
    for b, want in keep.items():
        assert torch.equal(bk.k_pool[1][b], want), f"block {b} was disturbed"


def test_self_copy_is_a_noop():
    bk = backend()
    fill(bk, 3, 5.0)
    before = [bk.k_pool[i][3].clone() for i in range(bk.num_layers)]
    bk.copy_block(3, 3)
    for i, want in enumerate(before):
        assert torch.equal(bk.k_pool[i][3], want)


@pytest.mark.parametrize("src,dst", [(-1, 0), (0, -1), (999, 0), (0, 999)])
def test_out_of_range_raises(src, dst):
    with pytest.raises(IndexError):
        backend().copy_block(src, dst)


def test_radix_cache_refuses_cow_without_the_callable():
    """
    The cache must RAISE rather than repoint without copying. Repointing would
    hand a sequence a block whose prefix KV was never written — fluent output,
    wrong attention, no error.
    """
    from serving.cache.radix import RadixCache
    from serving.memory.allocator import BlockAllocator
    from serving.memory.block_table import SequenceBlocks

    alloc = BlockAllocator(num_blocks=8, block_size=4)
    cache = RadixCache(alloc, block_copy=None)

    # A sequence holding one PARTIALLY-FILLED block that someone else also
    # references — the only situation in which COW is legal, per
    # ensure_writable's own contract.
    seq = SequenceBlocks(alloc, seq_id=0)
    seq.append(2)                                # 2 of 4 slots used
    alloc.incref(seq.block_ids)                  # a second holder => refcount 2

    with pytest.raises(RuntimeError, match="copy-on-write"):
        cache.ensure_writable(seq, 0)


def test_backends_expose_the_same_seam():
    from serving.backends.flashinfer_backend import FlashInferBackend
    assert callable(PagedTorchBackend.copy_block)
    assert callable(FlashInferBackend.copy_block)
