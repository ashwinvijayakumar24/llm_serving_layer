"""
INDEPENDENT VERIFICATION of the claim that batched GEMM shape changes logits.

The claim under test (from the P4 investigation): identical tokens at identical
positions produce DIFFERENT logits depending only on how many OTHER requests'
tokens shared the same forward pass. If true, batch invariance does not hold
exactly, and the P4 radix gate was failing for a reason that has nothing to do
with the radix cache.

This deliberately does NOT use the radix cache, the scheduler, or preemption.
It calls the engine's batched forward directly, twice, with the SAME target
sequence and different amounts of unrelated padding traffic beside it.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.loader import load_config, load_weights_gpu
from engine.model_gpu import LlamaModelGPU

from serving.backends.paged_torch import PagedTorchBackend
from serving.engine_iface.batch import ScheduledSeq, build_batch_meta, build_token_tensor
from serving.memory.allocator import BlockAllocator
from serving.memory.block_table import SequenceBlocks

W = os.environ.get("LLM_WEIGHTS_PATH", "vendor/llm_inference_engine/weights")
BS = 16
TARGET = [128000] + [9906] * 104          # 105 tokens
FILLER = [128000] + [3923] * 63           # 64 tokens each


def run(model, cfg, n_filler):
    """Prefill TARGET alongside n_filler unrelated sequences. Return its logits."""
    nb = 4096
    alloc = BlockAllocator(num_blocks=nb, block_size=BS)
    backend = PagedTorchBackend(
        num_layers=cfg["num_hidden_layers"], num_blocks=nb, block_size=BS,
        n_kv_heads=cfg["num_key_value_heads"], n_heads=cfg["num_attention_heads"],
        head_dim=cfg["head_dim"], device=model.device, dtype=torch.float16,
    )
    seqs = []
    for i, ids in enumerate([TARGET] + [FILLER] * n_filler):
        b = SequenceBlocks(alloc, seq_id=i)
        b.append(len(ids))
        seqs.append(ScheduledSeq(blocks=b, new_token_ids=list(ids)))
    meta = build_batch_meta(seqs, device=model.device, page_size=BS)
    toks = build_token_tensor(seqs, device=model.device)
    logits = model.forward_varlen(toks, meta, backend)
    return logits[0].float().cpu()          # TARGET is sequence 0 in both runs


def main():
    cfg = load_config(W)
    model = LlamaModelGPU(load_weights_gpu(W, cfg), cfg)

    base = run(model, cfg, 0)               # TARGET alone
    print(f"target alone: {len(TARGET)} tokens, argmax={int(base.argmax())}")
    print()
    hdr = (f"{'filler seqs':>12} {'batch tokens':>13} {'max|dlogit|':>12} "
           f"{'argmax':>8} {'flipped':>8}")
    print(hdr)
    flips = 0
    for n in (1, 2, 4, 6):
        other = run(model, cfg, n)
        d = float((other - base).abs().max())
        am = int(other.argmax())
        flip = am != int(base.argmax())
        flips += flip
        print(f"{n:>12} {len(TARGET)+n*len(FILLER):>13} {d:>12.5f} {am:>8} {str(flip):>8}")

    print()
    if flips:
        print("VERIFIED: argmax FLIPPED — batch composition changes the sampled token.")
    else:
        print("Logit drift measured above. Argmax did not flip in these cases;")
        print("drift magnitude is what determines whether it can.")
    print("The target sequence is byte-identical in every run. Only the number of")
    print("UNRELATED tokens sharing the forward pass differs.")


if __name__ == "__main__":
    main()
