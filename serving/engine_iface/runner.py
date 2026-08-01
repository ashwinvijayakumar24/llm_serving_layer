"""
Minimal batch-1 paged generation loop.

WHAT THIS IS FOR
----------------
This is the smallest thing that exercises the whole Phase 1 stack end to end:

    BlockAllocator -> SequenceBlocks -> build_batch_meta -> PagedTorchBackend
                                                         -> LlamaModelGPU.forward_varlen

Its job is to make the Phase 1 correctness claim testable — greedy output
through the paged path must be BIT-IDENTICAL to the engine's contiguous
reference path. Without a loop tying the pieces together, every component can be
individually correct and the composition still wrong.

WHAT THIS IS NOT
----------------
Not a scheduler. Batch 1, no queue, no admission control, no preemption, no
continuous batching — Phase 2 replaces it entirely. Deliberately so: keeping
batching out of Phase 1 means a paged-attention bug cannot hide behind a
batching bug, which is the only reason a divergence here is diagnosable.

It is written to be read alongside `engine/scheduler.py:generate()`, whose
structure it mirrors on purpose: prefill, sample, then decode one token at a
time. The differences are exactly the paged parts.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import torch

from serving.engine_iface.batch import ScheduledSeq, build_batch_meta, build_token_tensor
from serving.memory.block_table import SequenceBlocks

if TYPE_CHECKING:
    from serving.memory.allocator import BlockAllocator

__all__ = ["paged_generate", "PagedSession"]

# Matches engine/model_gpu.py:19. Duplicated rather than imported so the stop
# condition is visible at the point it is used; asserted equal in tests.
EOS_IDS = {128001, 128008, 128009}


class PagedSession:
    """
    One sequence's lifetime on the paged path: its block table, its token
    history, and the batch assembly that drives it.

    Exists as a class rather than locals inside the loop so a test can inspect
    block-table state mid-generation — which is how you tell "the tokens are
    wrong" apart from "the tokens are right but the memory accounting is not".
    """

    def __init__(self, allocator: BlockAllocator, model, backend, seq_id: int = 0):
        self.allocator = allocator
        self.model = model
        self.backend = backend
        self.blocks = SequenceBlocks(allocator, seq_id=seq_id)
        self.token_ids: list[int] = []
        self.device = model.device

    def _step(self, new_tokens: list[int]) -> torch.Tensor:
        """
        Run one forward pass over `new_tokens` and return (vocab,) logits for the
        final position.

        The four lines that matter, in order:
          1. grow the block table  (allocates physical blocks if needed)
          2. assemble BatchMeta    (positions, CSR page table, slot mapping)
          3. forward               (backend appends K/V then attends)
          4. logits for the last token

        Note step 1 happens BEFORE assembly: `build_batch_meta` reads kv_len and
        the page list off the block table, so the table must already reflect this
        step's tokens. Assembling first would describe a shorter sequence than
        the one about to be written, and the KV for the final tokens would land
        outside the pages the metadata names.
        """
        self.blocks.append(len(new_tokens))
        self.token_ids.extend(new_tokens)

        seq = ScheduledSeq(blocks=self.blocks, new_token_ids=new_tokens)
        meta = build_batch_meta([seq], device=self.device, page_size=self.allocator.block_size)
        tokens = build_token_tensor([seq], device=self.device)

        logits = self.model.forward_varlen(tokens, meta, self.backend)  # (1, vocab)
        return logits[0]

    def prefill(self, prompt_ids: list[int]) -> torch.Tensor:
        if self.token_ids:
            raise RuntimeError("prefill() called on a session that already has tokens")
        return self._step(list(prompt_ids))

    def decode(self, token_id: int) -> torch.Tensor:
        return self._step([token_id])

    def free(self) -> None:
        self.blocks.free()


def paged_generate(
    model,
    backend,
    allocator: BlockAllocator,
    prompt_ids: list[int],
    sampler_fn: Callable[[torch.Tensor], int],
    max_tokens: int = 64,
    seq_id: int = 0,
) -> Generator[int, None, None]:
    """
    Greedy/sampled generation over the paged path. Mirrors
    `engine/scheduler.py:generate()` so the two can be compared directly.

    Yields each generated token id, including EOS if produced. Blocks are freed
    on exhaustion AND on early termination — a generator abandoned mid-stream
    (client disconnect) would otherwise leak its entire block table, and a
    disconnect-heavy workload would drain the pool with no error anywhere. The
    engine's own loop cannot do this: it constructs a fresh KVCacheGPU per call
    and relies on garbage collection (engine/scheduler.py:26-27).

    `sampler_fn` takes a (vocab,) tensor and returns a token id, matching the
    engine's sampler contract except that the tensor is on-device rather than
    CPU numpy — forward_varlen deliberately keeps logits on the GPU.
    """
    session = PagedSession(allocator, model, backend, seq_id=seq_id)
    try:
        logits = session.prefill(list(prompt_ids))
        next_id = sampler_fn(logits)
        yield next_id
        if next_id in EOS_IDS:
            return

        for _ in range(max_tokens - 1):
            logits = session.decode(next_id)
            next_id = sampler_fn(logits)
            yield next_id
            if next_id in EOS_IDS:
                return
    finally:
        # Runs on normal exit, on early return, AND on generator close/abandon.
        session.free()


def greedy_device(logits: torch.Tensor) -> int:
    """
    Greedy sampler over an on-device logits tensor.

    Equivalent to engine/sampler.py:greedy but without the device->host copy.
    Bit-identical selection: argmax over the same fp16 values either way, since
    argmax is order-preserving under the copy.
    """
    return int(torch.argmax(logits).item())
