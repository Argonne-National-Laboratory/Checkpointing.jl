module Checkpointing

using LinearAlgebra
using ReverseDiff
using Zygote

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

function jacobian(tobedifferentiated, F_H, ::ReverseDiffADTool)
    return ReverseDiff.jacobian(tobedifferentiated, F_H)
end

function jacobian(tobedifferentiated, F_H, ::ZygoteADTool)
    return Zygote.jacobian(tobedifferentiated, F_H)[1]
end

function jacobian(tobedifferentiated, F_H, ::DiffractorADTool)
    return Zygote.jacobian(tobedifferentiated, F_H)[1]
end

include("Schemes/Revolve.jl")
include("Schemes/Periodic.jl")

export Revolve, guess, factor, next_action!, ActionFlag, Periodic
export ReverseDiffADTool, ZygoteADTool, jacobian

end
