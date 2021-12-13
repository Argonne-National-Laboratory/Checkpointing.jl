module Checkpointing

using LinearAlgebra
# All AD tools
using Diffractor, ForwardDiff, Enzyme, ReverseDiff, Zygote

export mynorm

abstract type Scheme end
"""
    Action

    none: no action
    store: store a checkpoint now equivalent to TAKESHOT in Alg. 79
    restore: restore a checkpoint now equivalent to RESTORE in Alg. 79
    forward: execute iteration(s) forward equivalent to ADVANCE in Alg. 79
    firstuturn: tape iteration(s); optionally leave to return later;  and (upon return) do the adjoint(s) equivalent to FIRSTTURN in Alg. 799
    uturn: tape iteration(s) and do the adjoint(s) equivalent to YOUTURN in Alg. 79
    done: we are done with adjoining the loop equivalent to the `terminate` enum value in Alg. 79
"""
@enum ActionFlag begin
    none
	store
	restore
	forward
	firstuturn
	uturn
	done
end

struct Action
	actionflag::ActionFlag
	iteration::Int
	startiteration::Int
	cpnum::Int
end

abstract type AbstractADTool end

struct ReverseDiffADTool <: AbstractADTool end
struct ZygoteADTool <: AbstractADTool end
struct DiffractorADTool <: AbstractADTool end
struct EnzymeADTool <: AbstractADTool end
struct ForwardDiffADTool <: AbstractADTool end

function jacobian(tobedifferentiated, F_H, ::ReverseDiffADTool)
    return ReverseDiff.jacobian(tobedifferentiated, F_H)
end

function jacobian(tobedifferentiated, F_H, ::ZygoteADTool)
    return Zygote.jacobian(tobedifferentiated, F_H)[1]
end

function jacobian(tobedifferentiated, F_H, ::ForwardDiffADTool)
    return ForwardDiff.jacobian(tobedifferentiated, F_H)
end

function jacobian(tobedifferentiated, F_H, ::EnzymeADTool)
    function f(x,res)
        y = tobedifferentiated(x)
        copyto!(res,y)
        return nothing
    end
    J = zeros(eltype(F_H), length(F_H), length(F_H))
    x = zeros(eltype(F_H), length(F_H))
    dx = zeros(eltype(F_H), length(F_H))
    y = zeros(eltype(F_H), length(F_H))
    dy = zeros(eltype(F_H), length(F_H))
    for i in 1:length(F_H)
        copyto!(x, F_H)
        fill!(dx, 0)
        fill!(y, 0)
        dy[i] = 1.0
        autodiff(f, Duplicated(x,dx), Duplicated(y, dy))
        J[i,:] = dx[:]
    end
    return J
end

function jacobian(tobedifferentiated, F_H, ::DiffractorADTool)
    J = zeros(eltype(F_H), length(F_H), length(F_H))
    for i in 1:length(F_H)
        grad = Diffractor.gradient(x -> tobedifferentiated(x)[i], F_H)
        J[i,:] = grad[:][1]
    end
    return J
end

include("Schemes/Revolve.jl")
include("Schemes/Periodic.jl")

export Revolve, guess, factor, next_action!, ActionFlag, Periodic
export ReverseDiffADTool, ZygoteADTool, EnzymeADTool, ForwardDiffADTool, DiffractorADTool, jacobian

end
