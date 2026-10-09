#!/bin/bash
# Create one isolated Python environment per competitor engine for the
# cross-engine study (docs/xengine/SPEC.md).
#
# WHY SEPARATE ENVS: vLLM and SGLang each pin their own torch (and often their own
# flashinfer / triton). Installing either into our `llm` env would replace the
# torch our server was validated against, and installing both into one env makes
# pip pick a torch that at least one of them was not built for. So:
#
#   our env      (conda `llm`)          harness client + our server
#   xengine-vllm  ($ENV_ROOT/xengine-vllm)   vLLM server only
#   xengine-sglang ($ENV_ROOT/xengine-sglang) SGLang server only
#
# The Makefile picks the server interpreter per arm through VLLM_PY / SGLANG_PY.
#
# Usage (on a PACE login or compute node, after `module load anaconda3`):
#   bash scripts/xengine/setup_envs.sh            # create both
#   bash scripts/xengine/setup_envs.sh vllm       # just one
#
# Re-running with an existing env reinstalls the pinned version into it.

set -euo pipefail

# PIN: confirm latest stable — see docs/xengine/SOURCE_NOTES.md
# These values have NOT been checked against the current releases. The source
# citations in SOURCE_NOTES.md are only valid for the exact version they were
# read at, so these must match it. Set XENGINE_PINS_CONFIRMED=1 once they do.
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
# PIN: confirm latest stable — see docs/xengine/SOURCE_NOTES.md
SGLANG_VERSION="${SGLANG_VERSION:-0.5.3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

ENV_ROOT="${ENV_ROOT:-$HOME/ps-simpliearn-0/xengine_envs}"
VERSIONS_FILE="${VERSIONS_FILE:-results/xengine/_env/engine_versions.txt}"

if [ "${XENGINE_PINS_CONFIRMED:-0}" != "1" ]; then
    echo "WARNING: engine version pins are UNCONFIRMED (vllm ${VLLM_VERSION}, sglang ${SGLANG_VERSION})."
    echo "         Check them against docs/xengine/SOURCE_NOTES.md, then rerun with"
    echo "         XENGINE_PINS_CONFIRMED=1 (or set VLLM_VERSION / SGLANG_VERSION)."
fi

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(vllm sglang)

mkdir -p "$ENV_ROOT" "$(dirname "$VERSIONS_FILE")"

make_env () {   # $1 = env prefix path; prints the env's python
    local prefix=$1
    if [ ! -x "$prefix/bin/python" ]; then
        if command -v conda >/dev/null 2>&1; then
            conda create -y -q -p "$prefix" "python=${PYTHON_VERSION}" >&2
        else
            "python${PYTHON_VERSION}" -m venv "$prefix" >&2
        fi
    fi
    "$prefix/bin/python" -m pip install -q --upgrade pip >&2
    echo "$prefix/bin/python"
}

install_engine () {   # $1 = vllm|sglang
    local name=$1 py
    case "$name" in
        vllm)
            py=$(make_env "$ENV_ROOT/xengine-vllm")
            "$py" -m pip install "vllm==${VLLM_VERSION}"
            ;;
        sglang)
            py=$(make_env "$ENV_ROOT/xengine-sglang")
            "$py" -m pip install "sglang[all]==${SGLANG_VERSION}"
            ;;
        *) echo "unknown engine: $name (expected vllm or sglang)"; exit 2 ;;
    esac
    # Record what actually got installed, not what was requested: pip may
    # resolve a different torch / flashinfer than the pin implies.
    {
        echo "=== ${name}  ($(date -u +%Y-%m-%dT%H:%M:%SZ))  python: ${py}"
        echo "requested: ${name}==$([ "$name" = vllm ] && echo "$VLLM_VERSION" || echo "$SGLANG_VERSION")"
        echo "pins_confirmed: ${XENGINE_PINS_CONFIRMED:-0}"
        "$py" - <<'PY'
import importlib, platform
print("python", platform.python_version())
for mod in ("vllm", "sglang", "torch", "flashinfer", "triton", "transformers", "xformers"):
    try:
        m = importlib.import_module(mod)
        print(mod, getattr(m, "__version__", "?"))
    except Exception as e:  # noqa: BLE001 - record any import failure verbatim
        print(mod, f"(not importable: {type(e).__name__})")
try:
    import torch
    print("torch.version.cuda", torch.version.cuda)
except Exception:
    pass
PY
        echo
    } | tee -a "$VERSIONS_FILE"
    "$py" -m pip freeze > "$(dirname "$VERSIONS_FILE")/pip_freeze_${name}.txt"
}

for t in "${TARGETS[@]}"; do
    install_engine "$t"
done

echo
echo "Versions appended to ${VERSIONS_FILE}. Point the Makefile at the envs with:"
echo "  VLLM_PY=$ENV_ROOT/xengine-vllm/bin/python"
echo "  SGLANG_PY=$ENV_ROOT/xengine-sglang/bin/python"
