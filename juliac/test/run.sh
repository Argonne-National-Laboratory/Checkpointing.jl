#!/bin/bash
# Compares the schedules of libckptjl with Enzyme's C reference schemes:
#   ./run.sh <dir with enzyme/checkpoint.h> [julia]
set -eu
cd "$(dirname "$0")"
J=${2:-$HOME/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia}
JL=$(dirname "$(readlink -f "$J")")/..
cc -O1 -I"$1" schedcmp.c -L.. -lckptjl -Wl,-rpath,"$PWD/.." -Wl,-rpath,"$JL/lib" \
  -Wl,-rpath,"$JL/lib/julia" -o schedcmp
./schedcmp
