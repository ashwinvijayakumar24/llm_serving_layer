# Cross-engine benchmark study (docs/xengine/SPEC.md) plus test / lint shortcuts.
#
# Each engine lives in its own Python environment, because vLLM and SGLang each
# pin their own torch build. The harness CLIENT (bench.xengine.run: load
# generation, metrics, artifact writing) always runs in OUR env ($(OURS_PY)).
# The SERVER for an arm runs under the interpreter chosen by the arm's engine;
# the harness reads it from $XENGINE_SERVER_PY. Point the three variables at
# real interpreters, e.g. after scripts/xengine/setup_envs.sh:
#
#   make bench-all VLLM_PY=$HOME/envs/xengine-vllm/bin/python \
#                  SGLANG_PY=$HOME/envs/xengine-sglang/bin/python
#
# One cell:   make bench ARM=vllm WORKLOAD=W1          (ENGINE= is an alias for ARM=)
# Full study: make bench-all && make render
# Appendix:   make bench-w5

PY        ?= python3
OURS_PY   ?= $(PY)
VLLM_PY   ?= $(PY)
SGLANG_PY ?= $(PY)

ENGINE   ?=
ARM      ?= $(ENGINE)
WORKLOAD ?=
REPS     ?= 3
OUT      ?= results/xengine

# The matrix from SPEC.md "Engines and arms" x W1-W4. Override to run a subset,
# e.g. make bench-all ARMS="ours vllm sglang" WORKLOADS=W1.
ARMS      ?= ours ours-noprefix \
             vllm vllm-eager vllm-noprefix vllm-matched \
             sglang sglang-noradix sglang-nooverlap sglang-eager
WORKLOADS ?= W1 W2 W3 W4

# W5 appendix: our engine int8 vs fp16 on W1/W2. The fp16 side is the `ours`
# cells bench-all already produced (re-running them here would write extra reps
# into the same directory). SPEC does not name the int8 arm id yet; "ours-int8"
# must match whatever the harness registers.
W5_ARMS      ?= ours-int8
W5_WORKLOADS ?= W1 W2

# Server interpreter for an arm: vllm* -> VLLM_PY, sglang* -> SGLANG_PY, else ours.
server_py = $(if $(filter vllm%,$(1)),$(VLLM_PY),$(if $(filter sglang%,$(1)),$(SGLANG_PY),$(OURS_PY)))

PYTEST_MARKERS ?= not gpu and not slow and not engine
# The repo predates `ruff format`; only files written for this study are held to it.
FORMAT_PATHS   ?= bench/xengine tests/test_xengine_render.py

.PHONY: bench bench-all bench-w5 render test lint help

help:
	@echo "make bench ARM=<arm> WORKLOAD=<W1..W4> [REPS=3]   one cell of the matrix"
	@echo "make bench-all                                    every arm x W1-W4 (sequential)"
	@echo "make bench-w5                                     W5 appendix (ours int8 vs fp16)"
	@echo "make render                                       refresh docs/xengine/BENCHMARKS.md"
	@echo "make test | make lint"

bench:
	@if [ -z "$(ARM)" ] || [ -z "$(WORKLOAD)" ]; then \
		echo "usage: make bench ARM=<arm> WORKLOAD=<id> [REPS=3]  (ENGINE= works too)"; exit 2; fi
	XENGINE_SERVER_PY="$(call server_py,$(ARM))" \
		$(OURS_PY) -m bench.xengine.run --arm $(ARM) --workload $(WORKLOAD) \
		--reps $(REPS) --out $(OUT)

# Workload-outer, arm-inner: each workload's arms run back to back, so slow drift
# over a long allocation spreads across engines instead of landing on one.
# A failed cell does not stop the matrix; failures are listed and the target
# exits non-zero at the end.
bench-all:
	@failed=""; \
	for wl in $(WORKLOADS); do \
		for arm in $(ARMS); do \
			echo "=== bench $$arm $$wl ==="; \
			$(MAKE) --no-print-directory bench ARM=$$arm WORKLOAD=$$wl REPS=$(REPS) OUT=$(OUT) \
				|| failed="$$failed $$arm/$$wl"; \
		done; \
	done; \
	if [ -n "$$failed" ]; then echo "FAILED cells:$$failed"; exit 1; fi

bench-w5:
	@failed=""; \
	for wl in $(W5_WORKLOADS); do \
		for arm in $(W5_ARMS); do \
			echo "=== bench $$arm $$wl (W5 appendix) ==="; \
			$(MAKE) --no-print-directory bench ARM=$$arm WORKLOAD=$$wl REPS=$(REPS) OUT=$(OUT) \
				|| failed="$$failed $$arm/$$wl"; \
		done; \
	done; \
	if [ -n "$$failed" ]; then echo "FAILED cells:$$failed"; exit 1; fi

render:
	$(OURS_PY) -m bench.xengine.render --results $(OUT)

test:
	$(OURS_PY) -m pytest -m "$(PYTEST_MARKERS)" -q -p no:cacheprovider

lint:
	ruff check .
	ruff format --check $(FORMAT_PATHS)
