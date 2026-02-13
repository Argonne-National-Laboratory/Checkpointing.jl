# GPU version of the 1D heat equation example.
# Reuses the Heat struct and advance/sumheat from heat.jl via Adapt.jl.
# Requires a GPU backend, e.g.: `using Pkg; Pkg.add("CUDA")`
#
# Usage:
#   using CUDA
#   include("examples/heat_gpu.jl")
#   T, dT = heat_gpu(Revolve(100), 500)

using Checkpointing
using Enzyme
using Adapt

include("heat.jl")

function heat_gpu(scheme::Scheme, tsteps::Int; arraytype = nothing)
    if arraytype === nothing
        error("Pass an array type, e.g.: heat_gpu(scheme, tsteps; arraytype=CuArray)")
    end

    n = 100
    λ = 0.5

    # Create CPU structs, then adapt to GPU
    heat_model = adapt(arraytype, Heat(zeros(n), zeros(n), n, λ, tsteps))
    dheat = adapt(arraytype, Heat(zeros(n), zeros(n), n, λ, tsteps))

    # Boundary conditions
    heat_model.Tnext[1] = 20.0
    heat_model.Tnext[end] = 0.0

    autodiff(
        Enzyme.ReverseWithPrimal,
        sumheat,
        Duplicated(heat_model, dheat),
        Const(scheme),
        Const(tsteps),
    )

    return Array(heat_model.Tnext), Array(dheat.Tnext[2:(end-1)])
end
