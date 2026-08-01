"""Baselines. Definitions are load-bearing; see docs/BENCHMARK_METHODOLOGY.md §6.

B1 engine batch-1 over HTTP        -- the floor
B2 static batching                 -- the honest continuous-batching comparison
B3 contiguous per-sequence KV      -- the paged-cache capacity comparison
B4 round-robin routing             -- table stakes, NOT a result
B5 least-outstanding-requests      -- THE REAL ROUTING BASELINE
B6 vLLM                            -- shape/sanity only, never a throughput claim
"""
