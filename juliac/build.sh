#!/bin/bash
# Builds libckptjl.so: Checkpointing.jl's schemes behind Enzyme's checkpoint
# scheme ABI (enzyme/checkpoint.h), compiled with juliac --trim=safe.
#   ./build.sh [julia]      (Julia 1.12 or later; juliac ships with it)
set -eu
cd "$(dirname "$0")"
J=${1:-$HOME/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia}
JC=$(dirname "$(readlink -f "$J")")/../share/julia/juliac/juliac.jl
"$J" --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
"$J" --project=. "$JC" --output-lib libckptjl --experimental --trim=safe \
  --compile-ccallable CkptJuliac.jl
nm -D libckptjl.so | grep ' T enzyme_ckpt_jl_'
echo "link with: -L$PWD -lckptjl -Wl,-rpath,$PWD -Wl,-rpath,$(dirname "$(readlink -f "$J")")/../lib -Wl,-rpath,$(dirname "$(readlink -f "$J")")/../lib/julia"
