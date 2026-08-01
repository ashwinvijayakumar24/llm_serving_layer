"""Radix prefix cache: token trie, refcounting, LRU eviction, copy-on-write.

AUTHOR-WRITTEN. Phase 4.
Explainability gate: block-boundary truncation, why refcount AND LRU are both
needed, when COW triggers.
"""
