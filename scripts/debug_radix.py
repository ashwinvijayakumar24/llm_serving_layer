"""
R6 DIAGNOSTIC — where does the cache-on run's KV first disagree with cache-off?

Runs the failing `system` / `conversational` structures from tests/test_radix_gpu.py
twice (cache off, cache on) and fingerprints, per sequence, the layer-0 K vector at
EVERY logical position after every forward pass. A prefix cache that hands over the
wrong (or never-written) block shows up as the first position whose fingerprint
differs — which localises the bug to a block index rather than to a token index.

    python3 scripts/debug_radix.py system
"""

from __future__ import annotations

import os
import sys

import torch

from bench.workloads.generator import LengthSpec, WorkloadConfig, generate
from serving.cache.radix import RadixCache
from serving.memory.allocator import BlockAllocator
from serving.scheduler.scheduler import Request, Scheduler, SchedulerConfig

WEIGHTS_PATH = os.environ.get("LLM_WEIGHTS_PATH", "vendor/llm_inference_engine/weights")
BLOCK_SIZE = 16
NUM_BLOCKS = 2048
VOCAB_LIMIT = 32_000


class Recorder:
    """Wraps the model; after each forward, fingerprints every sequence's whole KV."""

    def __init__(self, model):
        self._model = model
        self.kv: dict[int, list[float]] = {}      # seq_id -> per-position fingerprint
        self.trace: dict[int, list[tuple]] = {}   # seq_id -> [(q, kv_len, pages)]

    def __getattr__(self, name):
        return getattr(self._model, name)

    @property
    def device(self):
        return self._model.device

    def forward_varlen(self, tokens, meta, backend):
        out = self._model.forward_varlen(tokens, meta, backend)
        self._record(meta, backend)
        return out

    def _record(self, meta, backend):
        bs = meta.page_size
        indptr = meta.kv_indptr.tolist()
        indices = meta.kv_indices.tolist()
        kv_lens = meta.kv_lens.tolist()
        q_lens = meta.query_lens.tolist()
        k_flat = backend.k_pool[0].view(-1, backend.n_kv_heads, backend.head_dim)
        for i, klen in enumerate(kv_lens):
            pages = indices[indptr[i] : indptr[i + 1]]
            slots = [pages[p // bs] * bs + (p % bs) for p in range(klen)]
            idx = torch.tensor(slots, dtype=torch.long, device=k_flat.device)
            fp = k_flat[idx].float().sum(dim=(1, 2)).cpu().tolist()
            sid = self._seq_ids[i]
            self.kv[sid] = fp
            self.trace.setdefault(sid, []).append((q_lens[i], klen, list(pages)))


def make_stack(model, config, *, cache, **cfg):
    from serving.backends.paged_torch import PagedTorchBackend

    allocator = BlockAllocator(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    backend = PagedTorchBackend(
        num_layers=config["num_hidden_layers"], num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE, n_kv_heads=config["num_key_value_heads"],
        n_heads=config["num_attention_heads"], head_dim=config["head_dim"],
        device=model.device, dtype=torch.float16,
    )
    rc = RadixCache(allocator, enabled=True) if cache else None
    rec = Recorder(model)
    sched = Scheduler(rec, backend, allocator, SchedulerConfig(**cfg), prefix_cache=rc)
    return allocator, rc, sched, rec


def run_staged(sched, rec, groups, max_tokens):
    out, seqmap, admit = {}, {}, {}
    real_select = sched._select_batch

    for group in groups:
        for rid, ids in group.items():
            sched.add_request(
                Request(request_id=rid, prompt_ids=list(ids),
                        max_tokens=max_tokens, ignore_eos=True)
            )
        while sched.has_work:
            batch, _ = real_select()
            rec._seq_ids = [r.blocks.seq_id for r in batch]
            for r in batch:
                seqmap[r.blocks.seq_id] = r.request_id
                admit.setdefault(
                    r.request_id,
                    (len(r.prefill_ids), r.prefill_pos, r.cached_blocks,
                     list(r.blocks.block_ids)),
                )
            sched.step()
        out.update({r.request_id: list(r.output_ids) for r in sched.finished})
    return out, seqmap, admit


def main():
    structure = sys.argv[1] if len(sys.argv) > 1 else "system"
    from engine.loader import load_config, load_weights_gpu
    from engine.model_gpu import LlamaModelGPU

    config = load_config(WEIGHTS_PATH)
    model = LlamaModelGPU(load_weights_gpu(WEIGHTS_PATH, config), config)

    wl = generate(WorkloadConfig(
        n_requests=12, structure=structure, sharing_rate=0.9, block_size=BLOCK_SIZE,
        seed=3,
        prompt=LengthSpec(dist="lognormal", mean=96, sigma=0.4, min_len=32, max_len=192),
        output=LengthSpec(dist="fixed", mean=8),
        shared_prefix_tokens=64, adversarial_common_tokens=64, n_shared_prefixes=2,
        max_turns=3, vocab_size=VOCAB_LIMIT, name=f"dbg-{structure}",
    ))
    groups = [{r.request_id: list(r.token_ids)} for r in wl.requests]
    cfg = dict(max_batch_size=8, max_prefill_tokens=128)

    _, _, off, rec_off = make_stack(model, config, cache=False, **cfg)
    exp, map_off, adm_off = run_staged(off, rec_off, groups, 8)

    _, rc, on, rec_on = make_stack(model, config, cache=True, **cfg)
    got, map_on, adm_on = run_staged(on, rec_on, groups, 8)

    print(f"\n===== {structure} =====")
    inv_off = {v: k for k, v in map_off.items()}
    inv_on = {v: k for k, v in map_on.items()}

    for rid in sorted(exp):
        if got[rid] == exp[rid]:
            continue
        div = next(i for i, (a, b) in enumerate(zip(exp[rid], got[rid])) if a != b)
        print(f"\n--- {rid}: output diverges at token {div}")
        print(f"    off: {exp[rid]}")
        print(f"    on : {got[rid]}")
        print(f"    admit off (prompt_len, prefill_pos, cached_blocks, blocks): {adm_off[rid]}")
        print(f"    admit on  (prompt_len, prefill_pos, cached_blocks, blocks): {adm_on[rid]}")
        a, b = rec_off.kv[inv_off[rid]], rec_on.kv[inv_on[rid]]
        print(f"    kv len off={len(a)} on={len(b)}")
        bad = [p for p in range(min(len(a), len(b))) if abs(a[p] - b[p]) > 1e-2]
        if bad:
            p = bad[0]
            print(f"    FIRST KV MISMATCH at position {p} (block {p // BLOCK_SIZE}), "
                  f"{len(bad)} of {min(len(a), len(b))} positions differ")
            print(f"    mismatching blocks: {sorted({q // BLOCK_SIZE for q in bad})}")
            print(f"    off[{p}]={a[p]:.4f}  on[{p}]={b[p]:.4f}")
        else:
            print("    KV IDENTICAL at every position — the bug is not the KV contents")
        print(f"    trace off: {rec_off.trace[inv_off[rid]][:4]}")
        print(f"    trace on : {rec_on.trace[inv_on[rid]][:4]}")

    print("\nsnapshot:", {k: v for k, v in rc.snapshot().items() if k != "definitions"})


if __name__ == "__main__":
    main()
