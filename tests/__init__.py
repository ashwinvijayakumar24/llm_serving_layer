"""Correctness gates. Each one exists because a specific failure is SILENT.

batch invariance      (R4)  -- alone vs in a mixed batch, bit-identical
preemption equality   (R3)  -- forced pressure vs unpreempted, bit-identical
cache on/off equality (R6)  -- prefix cache makes it faster AND wrong otherwise
allocator leak        (R7)  -- free list returns to initial count exactly
block-straddle        (R8)  -- slot_mapping off-by-one only shows at boundaries
backend differential  (R9)  -- PagedTorch vs FlashInfer, bit-identical
"""
