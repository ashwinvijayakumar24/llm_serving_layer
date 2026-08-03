"""Radix prefix cache: token trie, refcounting, LRU eviction, copy-on-write.

AUTHOR-WRITTEN. Phase 4.
Must be derivable from first principles: block-boundary truncation, why refcount AND LRU are both
needed, when COW triggers.
"""

from serving.cache.radix import (
    CacheStats,
    MatchResult,
    RadixCache,
    RadixNode,
    attach_prefix,
)

__all__ = [
    "CacheStats",
    "MatchResult",
    "RadixCache",
    "RadixNode",
    "attach_prefix",
]
