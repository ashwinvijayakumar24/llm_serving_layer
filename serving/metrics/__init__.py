"""Metric definitions and exporters.

Every metric carries (quantity, unit, source) so a name cannot silently mean
two things -- the engine's known gap #1, where `peak_mem_mb` meant host RSS in
one harness and CUDA max_memory_allocated in another (BENCHMARKS.md:247).
"""
