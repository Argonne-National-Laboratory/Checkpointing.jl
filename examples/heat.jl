# Explicit 1D heat equation
using Checkpointing
using Enzyme

mutable struct Heat
    Tnext::Vector{Float64}
    Tlast::Vector{Float64}
    n::Int
    λ::Float64
    tsteps::Int
end

function advance(heat::Heat)
    next = heat.Tnext
    last = heat.Tlast
    λ = heat.λ
    n = heat.n
    for i = 2:(n-1)
        next[i] = last[i] + λ * (last[i-1] - 2 * last[i] + last[i+1])
    end
    return nothing
end


function sumheat_for(heat::Heat, chkpscheme::Scheme, tsteps::Int64)
    # AD: Create shadow copy for derivatives
    @checkpoint_struct chkpscheme heat for i = 1:tsteps
        # checkpoint_struct_for(advance, heat)
        heat.Tlast .= heat.Tnext
        advance(heat)
    end
    return reduce(+, heat.Tnext)
end

function sumheat_while(heat::Heat, chkpscheme::Scheme, tsteps::Int64)
    # AD: Create shadow copy for derivatives
    heat.tsteps = 1
    @checkpoint_struct chkpscheme heat while heat.tsteps <= tsteps
        heat.Tlast .= heat.Tnext
        advance(heat)
        heat.tsteps += 1
    end
    return reduce(+, heat.Tnext)
end

function heat_for(scheme::Scheme, tsteps::Int, ::EnzymeTool)
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
        sumheat_for,
        Duplicated(heat, dheat),
        Const(scheme),
        Const(tsteps),
    )

    return heat.Tnext, dheat.Tnext[2:end-1]
end

function heat_for(scheme::Scheme, tsteps::Int, ::ZygoteTool)
    n = 100
    Δx = 0.1
    Δt = 0.001
    # Select μ such that λ ≤ 0.5 for stability with μ = (λ*Δt)/Δx^2
    λ = 0.5

    # Create object from struct. tsteps is not needed for a for-loop
    heat = Heat(zeros(n), zeros(n), n, λ, tsteps)

    # Boundary conditions
    heat.Tnext[1] = 20.0
    heat.Tnext[end] = 0

    # Compute gradient
    g = Zygote.gradient(sumheat_for, heat, scheme, tsteps)

    return heat.Tnext, g[1].Tnext[2:end-1]
end

function heat_while(scheme::Scheme, tsteps::Int, ::EnzymeTool)
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
        sumheat_while,
        Duplicated(heat, dheat),
        Const(scheme),
        Const(tsteps),
    )

    return heat.Tnext, dheat.Tnext[2:end-1]
end

function heat_while(scheme::Scheme, tsteps::Int, ::ZygoteTool)
    n = 100
    Δx = 0.1
    Δt = 0.001
    # Select μ such that λ ≤ 0.5 for stability with μ = (λ*Δt)/Δx^2
    λ = 0.5

    # Create object from struct
    heat = Heat(zeros(n), zeros(n), n, λ, 1)

    # Boundary conditions
    heat.Tnext[1] = 20.0
    heat.Tnext[end] = 0

    # Compute gradient
    g = Zygote.gradient(sumheat_while, heat, scheme, tsteps)

    return heat.Tnext, g[1].Tnext[2:end-1]
end
