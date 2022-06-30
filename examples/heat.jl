# Explicit 1D heat equation
using Checkpointing
using Zygote

mutable struct Heat
    Tnext::Vector{Float64}
    Tlast::Vector{Float64}
    n::Int
    λ::Float64
    tsteps::Int
end

function advance(heat)
    next = heat.Tnext
    last = heat.Tlast
    λ = heat.λ
    n = heat.n
    for i in 2:(n-1)
        next[i] = last[i] + λ*(last[i-1]-2*last[i]+last[i+1])
    end
    return nothing
end


function sumheat(heat::Heat, chkpscheme::Scheme)
    # AD: Create shadow copy for derivatives
    @checkpoint_struct chkpscheme heat for i in 1:tsteps
        heat.Tlast .= heat.Tnext
        advance(heat)
    end
    return reduce(+, heat.Tnext)
end

function heat(scheme::Scheme, tsteps::Int)
    n = 100
    Δx=0.1
    Δt=0.001
    # Select μ such that λ ≤ 0.5 for stability with μ = (λ*Δt)/Δx^2
    λ = 0.5

    # Create object from struct
    heat = Heat(zeros(n), zeros(n), n, λ, tsteps)

    # Boundary conditions
    heat.Tnext[1]   = 20.0
    heat.Tnext[end] = 0

    # Compute gradient
    g = Zygote.gradient(sumheat, heat, scheme)

    return heat.Tnext, g[1].Tnext[2:end-1]
end
