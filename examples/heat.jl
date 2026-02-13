# Explicit 1D heat equation
using Checkpointing
using Enzyme
using Adapt

mutable struct Heat{T}
    Tnext::T
    Tlast::T
    n::Int
    λ::Float64
    tsteps::Int
end

function Adapt.adapt_structure(to, heat::Heat)
    return Heat(adapt(to, heat.Tnext), adapt(to, heat.Tlast), heat.n, heat.λ, heat.tsteps)
end

function advance(heat::Heat)
    heat.Tnext[2:(end-1)] .=
        heat.Tlast[2:(end-1)] .+
        heat.λ .* (heat.Tlast[1:(end-2)] .- 2 .* heat.Tlast[2:(end-1)] .+ heat.Tlast[3:end])
    return nothing
end


function sumheat(heat::Heat, scheme::Union{Revolve,Periodic}, tsteps::Int64)
    # AD: Create shadow copy for derivatives
    @ad_checkpoint scheme for i = 1:tsteps
        # checkpoint_struct_for(advance, heat)
        heat.Tlast .= heat.Tnext
        advance(heat)
    end
    return reduce(+, heat.Tnext)
end

function sumheat(heat::Heat, scheme::Online_r2, tsteps::Int64)
    # AD: Create shadow copy for derivatives
    heat.tsteps = 1
    @ad_checkpoint scheme while heat.tsteps <= tsteps
        heat.Tlast .= heat.Tnext
        advance(heat)
        heat.tsteps += 1
    end
    return reduce(+, heat.Tnext)
end

function heat(scheme::Scheme, tsteps::Int)
    n = 100
    Δx = 0.1
    Δt = 0.001
    # Select μ such that λ ≤ 0.5 for stability with μ = (λ*Δt)/Δx^2
    λ = 0.5

    # Create object from struct. tsteps is not needed for a for-loop
    heat = Heat(zeros(n), zeros(n), n, λ, tsteps)
    # Shadow copy for Enzyme
    dheat = Heat(zeros(n), zeros(n), n, λ, tsteps)

    # Boundary conditions
    heat.Tnext[1] = 20.0
    heat.Tnext[end] = 0

    # Compute gradient
    autodiff(
        Enzyme.ReverseWithPrimal,
        sumheat,
        Duplicated(heat, dheat),
        Const(scheme),
        Const(tsteps),
    )

    return heat.Tnext, dheat.Tnext[2:(end-1)]
end
