module Checkpointing

using ChainRulesCore
using LinearAlgebra
using Enzyme

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

function jacobian(tobedifferentiated, F_H, ::AbstractADTool)
    error("No AD tool interface implemented")
end

export AbstractADTool, jacobian, @checkpoint, @checkpoint_struct, checkpoint_struct

include("deprecated.jl")
include("Schemes/Revolve.jl")
include("Schemes/Periodic.jl")
include("Schemes/Online_r2.jl")

export Revolve, guess, factor, next_action!, ActionFlag, Periodic
export ReverseDiffADTool, ZygoteADTool, EnzymeADTool, ForwardDiffADTool, DiffractorADTool, jacobian

@generated function copyto!(dest::MT, src::MT) where {MT}
    assignments = [
        :( dest.$name = src.$name ) for name in fieldnames(MT)
    ]
    quote $(assignments...) end
end

function iszerotangent(tangent::TT) where {TT}
    isa(tangent, ZeroTangent)
end

@generated function copyto!(dest::MT, src::TT) where {MT,TT}
    ex = quote
    end
    ex
    assignments = [
        :( dest.$name = src.$name ) for name in fieldnames(MT) if iszerotangent(src.name)
    ]
    ex = quote 
        $(assignments...) 
    end
    return ex
end

to_named_tuple(p) = (; (v=>getfield(p, v) for v in fieldnames(typeof(p)))...)

function create_tangent(shadowmodel::MT) where {MT}
    shadowtuple = to_named_tuple(shadowmodel)
    return Tangent{MT,typeof(shadowtuple)}(shadowtuple) 
end

function ChainRulesCore.rrule(
    ::typeof(Checkpointing.checkpoint_struct),
    body::Function,
    alg::Scheme,
    model::MT,
    shadowmodel::MT,
) where {MT}
    model_input = deepcopy(model)
    for i in 1:alg.steps
        body(model)
    end
    function checkpoint_struct_pullback(dmodel)
        copyto!(shadowmodel, dmodel)
        model = checkpoint_struct(body, alg, model_input, shadowmodel)
        dshadowmodel = create_tangent(shadowmodel)
        return NoTangent(), NoTangent(), NoTangent(), dshadowmodel, NoTangent()
    end
    return model, checkpoint_struct_pullback
end

macro checkpoint_struct(alg, model, shadowmodel, loop)
    ex = quote
        $model = checkpoint_struct($alg, $model, $shadowmodel) do $model
            $(loop.args[2])
        end
    end
    esc(ex)
end

function checkpoint_struct(body::Function, alg, model_input::MT, shadowmodel::MT) where {MT}
    error("No checkpointing scheme implemented for algorithm $(typeof(alg)).")
end

end
