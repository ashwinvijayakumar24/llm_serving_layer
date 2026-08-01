"""Paged KV memory: block allocator, block tables, watermark policy.

AUTHOR-WRITTEN (docs/PRD.md §G7). Phase 1.
Explainability gate: alloc/free/refcount, slot_mapping vs block_tables,
why block-granularity, why the watermark exists.
"""
