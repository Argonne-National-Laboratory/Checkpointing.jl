# Checkpointed heat equation on the host and on a CUDA device, at growing sizes.
#
#     julia --project=test/gpu test/gpu/benchmark.jl
using CUDA
using Printf
include(joinpath(@__DIR__, "..", "..", "examples", "heat_gpu.jl"))

function best_time(arraytype, n, steps, snaps; samples = 2)
    heat_device(Revolve(snaps), steps; n = n, arraytype = arraytype)       # compile
    best = Inf
    for _ = 1:samples
        GC.gc(true)
        CUDA.synchronize()
        t = @elapsed begin
            heat_device(Revolve(snaps), steps; n = n, arraytype = arraytype)
            CUDA.synchronize()
        end
        best = min(best, t)
    end
    return best
end

CUDA.functional() || error("CUDA is not functional on this machine")
steps, snaps = 200, 20
println("Revolve($snaps), $steps steps, ", CUDA.name(CUDA.device()))
@printf("%10s %12s %12s %9s\n", "cells", "CPU (s)", "GPU (s)", "speedup")
for n in (10^3, 10^5, 10^6, 10^7)
    tc = best_time(Array, n, steps, snaps)
    tg = best_time(CuArray, n, steps, snaps)
    @printf("%10d %12.3f %12.3f %8.1fx\n", n, tc, tg, tc / tg)
end
