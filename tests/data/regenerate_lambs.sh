#!/usr/bin/env bash
# Regenerate Fortran-fk reference for the Lamb's problem test.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FK_SRC="${FK_SRC:-/tmp/py_cap/fk}"
[[ -x "$FK_SRC/fk" ]] || (cd "$FK_SRC" && make)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

# Vertical single force buried at depth 5 km in a near-homogeneous
# half-space (Lupei's fk requires source and receiver in different
# layers, so we have a 5 km "top crust" of identical properties).
# Receiver at the surface.
DISTANCES=(20 50 80 120)
TFIRST=(2 6 10 16)

{
    echo "2 2 1 1 0"
    echo "5.0   6.0 3.4 2.7 1000 500"
    echo "0.0   6.0 3.4 2.7 1000 500"
    echo "2 1024 0.05 0.5 25 1 1 1"
    echo "0. 1 0.3 15"
    echo "${#DISTANCES[@]}"
    for i in "${!DISTANCES[@]}"; do
        printf "%10.3f%10.3f l%s.\n" "${DISTANCES[$i]}" "${TFIRST[$i]}" "${DISTANCES[$i]}"
    done
} > input.fk

"$FK_SRC/fk" < input.fk > /dev/null

OUT_NPZ="${OUT_NPZ:-$HERE/lambs_halfspace.npz}"
WORKDIR="$WORK" OUT="$OUT_NPZ" python3 - <<'PY'
import os
from pathlib import Path
import numpy as np
from obspy import read

work = Path(os.environ["WORKDIR"])
out = Path(os.environ["OUT"]).resolve()
dists = [20, 50, 80, 120]
tfirst = [2, 6, 10, 16]
gf = []
for d in dists:
    rows = []
    # SF gives 6 components (n=0,1) × (Z, R, T)
    for s in "012345":
        tr = read(str(work / f"l{d}.{s}"))[0]
        rows.append(tr.data)
    gf.append(np.array(rows))
gf = np.array(gf)
np.savez_compressed(
    out,
    gf=gf.astype(np.float32),
    distances_km=np.array(dists, dtype=np.float64),
    tfirst_s=np.array(tfirst, dtype=np.float64),
    npts=1024, dt=0.05, sigma=2.0, taper=0.5, samples_before_p=25,
)
print("Wrote", out, gf.shape)
PY
