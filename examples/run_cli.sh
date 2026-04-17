#!/usr/bin/env bash
# Demonstrate the `fkpy compute` command-line interface.
#
# After `pip install -e .[dev]` the `fkpy` script is on $PATH.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# 1. Compute Green's functions for the canonical 5-layer model.
fkpy compute \
    --model "$HERE/zhu5.model" \
    --depth 8 \
    --distances 50,100,150 \
    --npts 2048 \
    --dt 0.1 \
    --workers 1 \
    --out "$HERE/greens.h5"

echo "Wrote $HERE/greens.h5"

# 2. Same run but also write per-receiver SAC files.
fkpy compute \
    --model "$HERE/zhu5.model" \
    --depth 8 \
    --distances 50 \
    --npts 1024 \
    --dt 0.2 \
    --workers 1 \
    --out "$HERE/greens_d50.h5" \
    --sac-prefix "$HERE/syn" \
    --azimuth 37.0 \
    --kstnm STA01

echo "SAC files:"
ls "$HERE"/syn.50.*.sac
