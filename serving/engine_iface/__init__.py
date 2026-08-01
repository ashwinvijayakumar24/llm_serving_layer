"""Adapters over the engine's `AttentionBackend` seam.

Owns `BatchMeta` construction (cu_query_lens, kv_lens, positions,
block_tables, slot_mapping, last_token_ix) and the varlen batch assembly
that `LlamaModelGPU.forward_varlen` consumes.

The protocol itself lives in the ENGINE (engine/attention_backend.py) so the
engine never imports from the serving layer. See docs/ARCHITECTURE.md §2.
"""
