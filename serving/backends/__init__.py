"""AttentionBackend implementations.

PagedTorchBackend  -- AUTHOR-WRITTEN. Pure PyTorch block-gather -> SDPA.
                      Written FIRST. Correctness oracle and hand-written reference.
                      Layout-independent, so a FlashInfer mismatch costs an
                      adapter rather than a redesign.
FlashInferBackend  -- integration glue. Fast path. NEVER claimed as authored.
                      Attribution wording is fixed in docs/PHASE_PLAN.md §11.
"""
