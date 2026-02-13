using Enzyme
using Adapt

mutable struct Model{T}
    F::T
    F_H::T
    t::Float64
    h::Float64
end

function Adapt.adapt_structure(to, model::Model)
    Model(adapt(to, model.F), adapt(to, model.F_H), model.t, model.h)
end

function func_U(t)
    e = exp(1)
    return 2.0 * ((e^(3.0 * t)) - (e^3)) / ((e^(3.0 * t / 2.0)) * (2.0 + (e^3)))
end

function func(F, X, t)
    u = func_U(t)
    x1 = X[1]
    F[2] = x1 * x1 + 0.5 * (u * u)
    F[1] = 0.5 * x1 + u
    return nothing
end

function advance(model)
    F_H = model.F_H
    F = model.F
    t = model.t
    h = model.h
    func(F, F_H, t)
    F .= F_H .+ (h / 2.0) .* F
    func(F, F, t + h / 2.0)
    model.F .= F_H .+ h .* F
    return nothing
end

function opt_sol(Y, t)
    e = exp(1)
    Y[1] = (2.0 * e^(3.0 * t) + e^3) / (e^(3.0 * t / 2.0) * (2.0 + e^3))
    Y[2] = (2.0 * e^(3.0 * t) - e^(6.0 - 3.0 * t) - 2.0 + e^6) / ((2.0 + e^3)^2)
    return
end

function opt_lambda(L, t)
    e = exp(1)
    L[1] = (2.0 * e^(3 - t) - 2.0 * e^(2.0 * t)) / (e^(t / 2.0) * (2 + e^3))
    L[2] = 1.0
    return
end
