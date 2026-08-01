"""
R6 DIAGNOSTIC — where does the cache-on run's KV first disagree with cache-off?

Runs a failing structure from tests/test_radix_gpu.py twice (cache off, cache on)
and fingerprints, per sequence, the layer-0 K vector at EVERY logical position
after every forward pass. A prefix cache that hands over the wrong (or
never-written) block shows up as a position whose fingerprint is wildly different;
fp16 reduction-order drift shows up as a position whose fingerprint differs in the
last bits. The two are told apart by MAGNITUDE, which is why the magnitudes are
printed rather than a boolean.

    python3 scripts/debug_radix.py system
"""

from __future__ import annotations

import os
import sys

import torch

import serving.scheduler.scheduler as sched_mod
from bench.workloads.generator import LengthSpec, WorkloadConfig, generate
from serving.cache.radix import RadixCache
from serving.memory.allocator import BlockAllocator
from serving.scheduler.scheduler import Request, Scheduler, SchedulerConfig

WEIGHTS_PATH = os.environ.get("LLM_WEIGHTS_PATH", "vendor/llm_inference_engine/weights")
BLOCK_SIZE = 16
NUM_BLOCKS = 2048
VOCAB_LIMIT = 32_000

_REAL_BUILD = sched_mod.build_batch_meta
_CURRENT: list = []


def _patched_build(seqs, device, page_size, **kw):
    _CURRENT[:] = [s.blocks.seq_id for s in seqs]
    return _REAL_BUILD(seqs, device, page_size, **kw)


sched_mod.build_batch_meta = _patched_build


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
        seq_ids = list(_CURRENT)
        out = self._model.forward_varlen(tokens, meta, backend)
        self._record(meta, backend, seq_ids)
        return out

    def _record(self, meta, backend, seq_ids):
        # EVERY layer, K and V. Layer 0's K is a function of (token, position)
        # alone — the embedding is the only input — so a layer-0 fingerprint is
        # blind to exactly the thing this diagnostic is looking for. Depth is
        # where context enters.
        bs = meta.page_size
        indptr = meta.kv_indptr.tolist()
        indices = meta.kv_indices.tolist()
        kv_lens = meta.kv_lens.tolist()
        q_lens = meta.query_lens.tolist()
        nl = backend.num_layers
        for i, klen in enumerate(kv_lens):
            pages = indices[indptr[i] : indptr[i + 1]]
            slots = [pages[p // bs] * bs + (p % bs) for p in range(klen)]
            idx = torch.tensor(slots, dtype=torch.long, device=backend.device)
            rows = []
            for lay in range(nl):
                for pool in (backend.k_pool[lay], backend.v_pool[lay]):
                    flat = pool.view(-1, backend.n_kv_heads, backend.head_dim)
                    rows.append(flat[idx].float().sum(dim=(1, 2)))
            fp = torch.stack(rows)                      # (2*layers, klen)
            sid = seq_ids[i]
            self.kv[sid] = fp.cpu()
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


def run_staged(sched, groups, max_tokens):
    out, seqmap, admit = {}, {}, {}
    for group in groups:
        for rid, ids in group.items():
            sched.add_request(
                Request(request_id=rid, prompt_ids=list(ids),
                        max_tokens=max_tokens, ignore_eos=True)
            )
        while sched.has_work:
            sched.step()
            for r in sched.running:
                if r.blocks is not None:
                    seqmap[r.blocks.seq_id] = r.request_id
                    admit.setdefault(
                        r.request_id,
                        (len(r.prefill_ids), r.prefill_pos, r.cached_blocks,
                         list(r.blocks.block_ids)),
                    )
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
    # A UNIFORM prefill budget makes every chunk exactly one block, for the
    # publisher and for the consumer, with the cache on and with it off. Every
    # forward pass that computes position p then packs the same number of
    # tokens and attends over the same kv_len, so if reuse is the only variable
    # left, the KV must be bit-identical. Pass 16 as argv[2] to run it.
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    cfg = dict(max_batch_size=8, max_prefill_tokens=budget)
    print(f"### structure={structure} max_prefill_tokens={budget}", flush=True)

    _, _, off, rec_off = make_stack(model, config, cache=False, **cfg)
    exp, map_off, adm_off = run_staged(off, groups, 8)

    _, rc, on, rec_on = make_stack(model, config, cache=True, **cfg)
    got, map_on, adm_on = run_staged(on, groups, 8)

    print(f"\n===== {structure} =====", flush=True)
    inv_off = {v: k for k, v in map_off.items()}
    inv_on = {v: k for k, v in map_on.items()}

    for rid in sorted(exp):
        same = got[rid] == exp[rid]
        a, b = rec_off.kv[inv_off[rid]], rec_on.kv[inv_on[rid]]
        n = min(a.shape[1], b.shape[1])
        d = (a[:, :n] - b[:, :n]).abs()
        per_pos = d.max(dim=0).values
        per_lay = d.max(dim=1).values
        nz = (per_pos > 0).nonzero().flatten().tolist()
        pl, pp, cb, blk = adm_on[rid]
        tag = "OK " if same else "DIV"
        npref = min(cb * BLOCK_SIZE, n)
        print(f"     [prefix drift {rid}: max|d| over [0,{npref}) = "
              f"{(d[:, :npref].max() if npref else 0):.5g}]", flush=True)
        print(f"{tag} {rid}: prompt={pl} prefill_pos_on={pp} cached_blocks={cb} "
              f"kv_len off={a.shape[1]} on={b.shape[1]} | KV max|d|={d.max():.4g} "
              f"n_differing_pos={len(nz)}/{n} first={nz[0] if nz else '-'} "
              f"first_layerpair={(per_lay > 0).nonzero().flatten().tolist()[:1]}", flush=True)
        if not same:
            div = next(i for i, (x, y) in enumerate(
                zip(exp[rid], got[rid], strict=False)) if x != y)
            print(f"     output diverges at token {div}")
            print(f"     off: {exp[rid]}")
            print(f"     on : {got[rid]}")
            print(f"     admit off: {adm_off[rid]}")
            print(f"     admit on : {(pl, pp, cb, blk)}")
            print(f"     first 40 differing positions: {nz[:40]}")
            print(f"     per-layerpair max|d|: "
                  f"{[round(float(x), 5) for x in per_lay.tolist()]}")
            print(f"     trace off: {rec_off.trace[inv_off[rid]][:3]}")
            print(f"     trace on : {rec_on.trace[inv_on[rid]][:3]}")
            cached_tokens = cb * BLOCK_SIZE
            npref = min(cached_tokens, n)
            if npref:
                print(f"     max|d| over the REUSED prefix [0,{npref}): "
                      f"{d[:, :npref].max():.5g}")
            if n > npref:
                print(f"     max|d| over the recomputed region [{npref},{n}): "
                      f"{d[:, npref:].max():.5g}")
            prompt = next(g[rid] for g in groups if rid in g)
            print("     SAME PROMPT, CACHE OFF, different prefill splits:")
            for budget in (1 << 20, cached_tokens or 64, 32, 16):
                _, _, s2, _ = make_stack(model, config, cache=False,
                                         max_batch_size=1, max_prefill_tokens=budget)
                s2.add_request(Request(request_id="x", prompt_ids=list(prompt),
                                       max_tokens=8, ignore_eos=True))
                s2.run_until_idle()
                print(f"       budget={budget:>7}: {s2.finished[0].output_ids}")

    print("\nsnapshot:", {k: v for k, v in rc.snapshot().items() if k != "definitions"})
    chunk_shape_selftest(model, config, groups)
    batch_shape_selftest(model, config, groups)


def chunk_shape_selftest(model, config, groups):
    """
    IS THE KV EVEN A FUNCTION OF THE TOKENS ALONE?

    Prefill the SAME prompt with two different chunk budgets, cache off both
    times, and compare the resulting K. If the two disagree, the KV depends on
    the SHAPE of the forward pass as well as on its content, and "reuse is
    bit-identical because it is the same tokens at the same positions by the same
    kernel" is false as stated — which is a claim about the engine, not about the
    radix cache.
    """
    print("\n===== chunk-shape self-test (cache OFF both runs) =====", flush=True)
    prompt = list(next(iter(groups[0].values())))
    fps = {}
    for budget in (256, 128, 48, 32):
        _, _, s, rec = make_stack(model, config, cache=False,
                                  max_batch_size=1, max_prefill_tokens=budget)
        s.add_request(Request(request_id="x", prompt_ids=list(prompt),
                              max_tokens=1, ignore_eos=True))
        s.run_until_idle()
        fps[budget] = rec.kv[0]
    base = fps[256]
    for budget, fp in fps.items():
        n = min(base.shape[1], fp.shape[1])
        d = (base[:, :n] - fp[:, :n]).abs()
        npos = int((d.max(dim=0).values > 0).sum())
        print(f"  prompt_len={len(prompt)} budget={budget}: max|d| vs budget=256 "
              f"is {d.max():.4g}, differing positions {npos}/{n}", flush=True)


def batch_shape_selftest(model, config, groups):
    """
    WHICH SHAPE IS THE KV SENSITIVE TO — the batch's, or the sequence's?

    Same prompt, same chunking (one chunk, whole prompt), cache off both times.
    The only difference is how many OTHER tokens share the forward pass, i.e.
    the M of every `linear()`. A nonzero difference means the packed batch width
    alone changes the KV, which is a statement about the engine and about the
    batch-invariance gate, not about the prefix cache. Zero means the
    sensitivity is per-sequence — `q_len` and `kv_len` inside `attend` — which
    is the axis prefix reuse necessarily moves.
    """
    print("\n===== batch-shape self-test (cache OFF both runs) =====", flush=True)
    prompts = [list(next(iter(g.values()))) for g in groups[:4]]
    target = prompts[0]

    _, _, s1, r1 = make_stack(model, config, cache=False,
                              max_batch_size=1, max_prefill_tokens=4096)
    s1.add_request(Request(request_id="solo", prompt_ids=list(target),
                           max_tokens=1, ignore_eos=True))
    s1.run_until_idle()
    solo = r1.kv[0]

    _, _, s2, r2 = make_stack(model, config, cache=False,
                              max_batch_size=8, max_prefill_tokens=4096)
    for i, p in enumerate(prompts):
        s2.add_request(Request(request_id=f"r{i}", prompt_ids=list(p),
                               max_tokens=1, ignore_eos=True))
    s2.run_until_idle()
    grouped = r2.kv[0]

    n = min(solo.shape[1], grouped.shape[1])
    d = (solo[:, :n] - grouped[:, :n]).abs()
    print(f"  target prompt_len={len(target)}, batched with "
          f"{sum(len(p) for p in prompts[1:])} other tokens in the same forward: "
          f"max|d|={d.max():.4g}, differing positions "
          f"{int((d.max(dim=0).values > 0).sum())}/{n}", flush=True)


if __name__ == "__main__":
    main()
