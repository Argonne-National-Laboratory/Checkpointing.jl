#!/bin/bash
# Compiles heatdrive.jl with juliac --trim=safe --output-exe and runs it: the
# loop-body snapshots of EnzymeLLVM, driven through its scheme table as
# Enzyme's driver drives them, restore exactly, for :all, :accessed, :written.
#   ./run.sh [julia]      (Julia 1.12 or later)
set -eu
cd "$(dirname "$0")"
J=${1:-$HOME/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia}
JC=$(dirname "$(readlink -f "$J")")/../share/julia/juliac/juliac.jl
"$J" --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
"$J" --project=. "$JC" --output-exe heatdrive --experimental --trim=safe heatdrive.jl
./heatdrive
