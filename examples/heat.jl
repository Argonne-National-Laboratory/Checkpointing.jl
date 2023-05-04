# Explicit 1D heat equation
using Enzyme
# using EnzymeCore
# import EnzymeCore.EnzymeRules: forward, augmented_primal, reverse, has_rrule_from_sig, has_frule_from_sig
using Checkpointing
# import EnzymeCore.EnzymeRules: forward, augmented_primal, reverse, has_rrule_from_sig, has_frule_from_sig
using Enzyme: EnzymeRules
import .EnzymeRules: inactive, augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules

# using Zygote

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


function sumheat(heat::Heat, chkpscheme::Scheme, tsteps::Int64)
    # AD: Create shadow copy for derivatives
    @checkpoint_struct chkpscheme heat for i in 1:tsteps
    # checkpoint_struct_for(advance, heat)
        heat.Tlast .= heat.Tnext
        @show heat.Tnext
        advance(heat)
    end
    @show heat.Tnext
    return reduce(+, heat.Tnext)
end

# function heat_for(scheme::Scheme, tsteps::Int)
#     n = 100
#     Δx=0.1
#     Δt=0.001
#     # Select μ such that λ ≤ 0.5 for stability with μ = (λ*Δt)/Δx^2
#     λ = 0.5

#     # Create object from struct. tsteps is not needed for a for-loop
#     heat = Heat(zeros(n), zeros(n), n, λ, tsteps)

#     # Boundary conditions
#     heat.Tnext[1]   = 20.0
#     heat.Tnext[end] = 0
#     #sumheat_for(heat, scheme, tsteps)

#     # Compute gradient
#     # g = Zygote.gradient(sumheat_for, heat, scheme, tsteps)
#     heat = Heat(zeros(n), zeros(n), n, λ, tsteps)
#     dheat = Heat(zeros(n), zeros(n), n, 0.0, 1)
#     @show typeof(Duplicated(heat, dheat))
#     autodiff(Enzyme.Reverse, sumheat_for, Active, Duplicated(heat, dheat), scheme, tsteps)

#     return heat, dheat
# end

function energy()
    tsteps = 10
    revolve = Revolve{Heat}(10, 5; verbose=1)
    n = 100
    Δx=0.1
    Δt=0.001
    # Select μ such that λ ≤ 0.5 for stability with μ = (λ*Δt)/Δx^2
    λ = 0.5

    # Create object from struct. tsteps is not needed for a for-loop
    heat = Heat(zeros(n), zeros(n), n, λ, tsteps)

    # Boundary conditions
    heat.Tnext[1]   = 20.0
    heat.Tnext[end] = 0
    #sumheat_for(heat, scheme, tsteps)

    # Compute gradient
    # g = Zygote.gradient(sumheat_for, heat, scheme, tsteps)

#     return heat, dheat
    T = sumheat(heat, revolve, tsteps)
    # println("Tnext = ", Tnext)
    # println("grad = ", grad)
end

function energy_adjoint()
    tsteps = 10
    revolve = Revolve{Heat}(10, 5; verbose=1)
    n = 100
    Δx=0.1
    Δt=0.001
    # Select μ such that λ ≤ 0.5 for stability with μ = (λ*Δt)/Δx^2
    λ = 0.5

    # Create object from struct. tsteps is not needed for a for-loop
    heat = Heat(zeros(n), zeros(n), n, λ, tsteps)
    dheat = Heat(zeros(n), zeros(n), n, λ, tsteps)

    # Boundary conditions
    heat.Tnext[1]   = 20.0
    heat.Tnext[end] = 0
    #sumheat_for(heat, scheme, tsteps)

    # Compute gradient
    # g = Zygote.gradient(sumheat_for, heat, scheme, tsteps)

#     return heat, dheat
    autodiff(Enzyme.Reverse, sumheat, Duplicated(heat, dheat), revolve, tsteps)
    # println("Tnext = ", Tnext)
    # println("grad = ", grad)
    return dheat
end

# function EnzymeRules.inactive(::typeof(checkpoint_struct_for), args...)
#     return nothing
# end

# if isinteractive()
    T = energy()
    dheat = energy_adjoint()
# end
methods(augmented_primal)

# x = [1.0]
# dx = [0.0]
# @show grad = autodiff(Reverse, g, Active, Duplicated(x, dx))
# @show dx