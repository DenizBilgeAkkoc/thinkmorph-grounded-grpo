#!/usr/bin/env bash
# Source after activating the Python environment. No host-specific paths or secrets.
THINKMORPH_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$THINKMORPH_REPO_ROOT/src:$THINKMORPH_REPO_ROOT/scripts:$THINKMORPH_REPO_ROOT/vendor/VLMEvalKit${PYTHONPATH:+:$PYTHONPATH}"
export LMUData="${LMUData:-$THINKMORPH_REPO_ROOT/data/benchmarks}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$THINKMORPH_REPO_ROOT/cache/matplotlib}"
mkdir -p "$LMUData" "$MPLCONFIGDIR"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPLIT_THINK=1 PRINT_VANILLA=1 VLMEVAL_FAIL_FAST=1
