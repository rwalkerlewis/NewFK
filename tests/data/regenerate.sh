#!/usr/bin/env bash
# Regenerate the frozen reference Green's functions used by the
# regression test, by running Lupei Zhu's Fortran `fk` against the
# canonical 5-layer model.
#
# Requires: gfortran, gcc, make, and the upstream `fk` source tree
# (e.g. as bundled in https://github.com/liuqinya/py_cap under fk/).
#
# Usage:
#   FK_SRC=/path/to/fk/source ./regenerate.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FK_SRC="${FK_SRC:-/tmp/py_cap/fk}"

if [[ ! -x "$FK_SRC/fk" ]]; then
    echo "Building Fortran fk in $FK_SRC ..." >&2
    (cd "$FK_SRC" && make)
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cd "$WORK"

# Build the fk input file.  Source layer index 4 (0-based 3) puts the
# source at depth 30 km (top of the 4th layer at 3+12+15=30 km).
# Receiver layer index 1 (0-based 0) puts it at the surface.
# DC source (stype=2), all-modes integration (updn=0).
DISTANCES=(50 80 100 130 160 200)
TFIRST=(7 12 15 19 24 30)

# All distances in a single fk run so that xmax (and therefore dk) match
# what compute_greens uses for the full batch.
{
    echo "5 4 2 1 0"
    echo "3.0   5.50 3.18 2.65 600  300"
    echo "12.0  6.30 3.64 2.78 800  400"
    echo "15.0  6.70 3.87 2.85 1000 500"
    echo "10.0  7.20 4.16 2.95 1500 800"
    echo "0.0   8.10 4.68 3.36 2000 1000"
    echo "2 1024 0.1 0.5 25 1 1 1"
    echo "0. 1 0.3 15"
    echo "${#DISTANCES[@]}"
    for i in "${!DISTANCES[@]}"; do
        printf "%10.3f%10.3f d%s.\n" "${DISTANCES[$i]}" "${TFIRST[$i]}" "${DISTANCES[$i]}"
    done
} > input.fk

"$FK_SRC/fk" < input.fk > /dev/null

# Pack into a single npz
OUT_NPZ="${OUT_NPZ:-$HERE/canonical_zhu5.npz}"
WORKDIR="$WORK" OUT="$OUT_NPZ" python3 - <<'PY'
import os
from pathlib import Path

import numpy as np
from obspy import read

work = Path(os.environ["WORKDIR"])
out_npz = Path(os.environ["OUT"]).resolve()

dists = [50, 80, 100, 130, 160, 200]
tfirst = [7, 12, 15, 19, 24, 30]
gf = []  # (n_dist, 9, 1024)
for d in dists:
    rows = []
    for s in "012345678":
        tr = read(str(work / f"d{d}.{s}"))[0]
        rows.append(tr.data)
    gf.append(np.array(rows))
gf = np.array(gf)

np.savez_compressed(
    out_npz,
    gf=gf.astype(np.float32),
    distances_km=np.array(dists, dtype=np.float64),
    tfirst_s=np.array(tfirst, dtype=np.float64),
    npts=1024,
    dt=0.1,
    sigma=2.0,
    taper=0.5,
    samples_before_p=25,
    src_depth_km=30.0,
    component_order="DD_Z DD_R DS_Z DS_R DS_T SS_Z SS_R SS_T",
)
print("Wrote", out_npz, "shape", gf.shape)
PY
