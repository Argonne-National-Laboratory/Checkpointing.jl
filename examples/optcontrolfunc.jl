using Enzyme
mutable struct Model
    F::Vector{Float64}
    F_H::Vector{Float64}
    t::Float64
    h::Float64
end

function func_U(t)
    e = exp(1)
    return 2.0 * ((e^(3.0 * t)) - (e^3)) / ((e^(3.0 * t / 2.0)) * (2.0 + (e^3)))
end

function func(F, X, t)
    F[2] = X[1] * X[1] + 0.5 * (func_U(t) * func_U(t))
    F[1] = 0.5 * X[1] + func_U(t)
    return nothing
end

function advance(model)
    F_H = model.F_H
    F = model.F
    t = model.t
    h = model.h
    func(F, F_H, t)
    F[1] = F_H[1] + h / 2.0 * F[1]
    F[2] = F_H[2] + h / 2.0 * F[2]
    func(F, F, t + h / 2.0)
    model.F[1] = F_H[1] + h * F[1]
    model.F[2] = F_H[2] + h * F[2]
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
