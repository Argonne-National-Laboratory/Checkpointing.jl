# The 1D heat equation of `heat.jl`, written so the same code runs on the CPU and
# on a GPU. Pass `arraytype = CuArray` (or `ROCArray`, `MtlArray`, ...) to run it
# on a device; checkpoints are then stored on the device too.
#
# Three constraints come from differentiating GPU broadcasts with Enzyme, not
# from Checkpointing.jl:
#
# 1. No floating-point scalars inside a broadcast. `x .= λ .* y` with a
#    `Float64` λ -- a struct field or even a literal `0.5` -- fails with
#    `MethodError: no method matching MixedDuplicated(::Broadcasted...)`. Keep
#    coefficients in device arrays instead, as `λ` is here. Integer literals
#    such as the `2` below are fine.
# 2. No views. Broadcasting over `@views` slices of a device array aborts
#    Enzyme (`number of arg operands != function parameters`). The stencil
#    below uses whole-array broadcasts with `circshift` instead.
# 3. No scalar indexing, which Enzyme cannot differentiate on a device.
#
# `λ` is zero at both ends, so the boundary cells keep their values and the
# wraparound terms `circshift` introduces are multiplied away -- no partial
# indexing needed.

using Checkpointing
using Enzyme
using Adapt

mutable struct DeviceHeat{A}
    Tnext::A
    Tlast::A
    λ::A
    tsteps::Int
end

function Adapt.adapt_structure(to, h::DeviceHeat)
    return DeviceHeat(adapt(to, h.Tnext), adapt(to, h.Tlast), adapt(to, h.λ), h.tsteps)
end

function advance!(h::DeviceHeat)
    L = h.Tlast
    h.Tnext .= L .+ h.λ .* (circshift(L, 1) .- 2 .* L .+ circshift(L, -1))
    return nothing
end

function sumheat(h::DeviceHeat, scheme::Union{Revolve,Periodic}, tsteps::Int)
    @ad_checkpoint scheme for i = 1:tsteps
        h.Tlast .= h.Tnext
        advance!(h)
    end
    return sum(h.Tnext)
end

function sumheat(h::DeviceHeat, scheme::Online_r2, tsteps::Int)
    h.tsteps = 1
    @ad_checkpoint scheme while h.tsteps <= tsteps
        h.Tlast .= h.Tnext
        advance!(h)
        h.tsteps += 1
    end
    return sum(h.Tnext)
end

"""
    heat_device(scheme, tsteps; n = 100, arraytype = Array) -> (T, dT)

Differentiate the heat equation with `scheme` and return the final temperature
and its gradient as host arrays. Set up on the host, then moved to `arraytype`.
"""
function heat_device(scheme::Scheme, tsteps::Int; n::Int = 100, arraytype = Array)
    T0 = zeros(n)
    T0[1] = 20.0
    λ = fill(0.5, n)
    λ[1] = λ[end] = 0.0
    h = adapt(arraytype, DeviceHeat(T0, zeros(n), λ, 0))
    dh = adapt(arraytype, DeviceHeat(zeros(n), zeros(n), zeros(n), 0))
    autodiff(
        Enzyme.Reverse,
        sumheat,
        Active,
        Duplicated(h, dh),
        Const(scheme),
        Const(tsteps),
    )
    return Array(h.Tnext), Array(dh.Tnext)[2:(end-1)]
end
