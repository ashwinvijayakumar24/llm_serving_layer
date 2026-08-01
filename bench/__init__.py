"""Open-loop load generation and benchmark harness.

Latency is measured from INTENDED DISPATCH TIME, never actual send time --
this is the coordinated-omission guard (docs/RISK_REGISTER.md R1), the most
likely way this project would publish a confidently wrong number.

Reused from the engine's bench/ as-is: _percentile, _hw_metadata,
write_results, _weight_mem_mb, _time_cuda.
Rewritten: everything else -- the engine's harness is closed-loop, single
stream, in-process, with no HTTP client anywhere.
"""
