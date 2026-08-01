"""Iteration-level scheduler: admission, continuous batching, preemption.

AUTHOR-WRITTEN. Phases 2-3.
Preemption implements BOTH recompute and swap behind a policy flag; the
comparison is the deliverable (docs/ARCHITECTURE.md §5.2).
"""
