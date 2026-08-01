"""Workload models: arrival process, length distributions, prefix sharing.

Four prefix-sharing structures, all swept (docs/BENCHMARK_METHODOLOGY.md §4):
  zero        -- the control. Measures what the cache COSTS when it never helps.
  system      -- small shared prefix set. The common production shape.
  conversational -- deep sharing. Most favorable; labeled as such.
  adversarial -- divergence straddling block boundaries. Where bugs live.

Realized distributions are published per run, not requested parameters (R14).
"""
