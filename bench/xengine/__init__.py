"""
Cross-engine benchmark harness: this serving layer vs vLLM vs SGLang.

The binding contract is docs/xengine/SPEC.md. This package produces one JSON
artifact per engine arm x workload x point x repetition, in the
`xengine-run/1` schema, and nothing else. Rendering lives in
`bench/xengine/render.py` (a separate deliverable) and reads only those files.

Modules:
  config     YAML configs (SLO, workloads) -> seeded request lists
  engines    one adapter per engine, the arm table, server process lifecycle
  closed_loop  W1's fixed-in-flight-count runner
  hardware   GPU / driver / CUDA / node capture, plus an optional clock sampler
  anomalies  bimodality, warmup, rep-to-rep spread, GPU clock drift detectors
  artifact   build + validate the xengine-run/1 artifact
  run        the CLI: `python3 -m bench.xengine.run --arm vllm --workload W1`
"""

SCHEMA = "xengine-run/1"
HARNESS_VERSION = "0.1.0"
