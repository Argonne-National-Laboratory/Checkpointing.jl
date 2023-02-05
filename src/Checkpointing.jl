module Checkpointing

using ChainRulesCore
using LinearAlgebra
using DataStructures
using Enzyme
using Serialization
using HDF5

"""
    Scheme

Abstract type from which all checkpointing schemes are derived.

"""
abstract type Scheme end

"""
    ActionFlag

Each checkpointing algorithm currently uses the same ActionFlag type for setting the next action in the checkpointing scheme
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
    err
	done
end

"""
    Action

Stores the state of the checkpointing scheme after an action is taken.
    * `actionflag` is the next action
    * `iteration` is number of iterations for move forward
    * `startiteration` is the loop step to start from
    * `cpnum` is the checkpoint index number

"""
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

export Scheme, AbstractADTool, jacobian
export @checkpoint, @checkpoint_struct, checkpoint_struct_for, checkpoint_struct_while
export reset

function serialize(x)
    s = IOBuffer()
    Serialization.serialize(s, x)
    take!(s)
end

function deserialize(x)
    s = IOBuffer(x)
    Serialization.deserialize(s)
end

export serialize, deserialize

abstract type AbstractStorage end

include("Storage/ArrayStorage.jl")
include("Storage/HDF5Storage.jl")

export AbstractStorage, ArrayStorage, HDF5Storage

include("deprecated.jl")
include("Schemes/Revolve.jl")
include("Schemes/Periodic.jl")
include("Schemes/Online_r2.jl")

export Revolve, guess, factor, next_action!, ActionFlag, Periodic
export ReverseDiffADTool, ZygoteADTool, EnzymeADTool, ForwardDiffADTool, DiffractorADTool, jacobian
export Online_r2, update_revolve

@generated function copyto!(dest::MT, src::MT) where {MT}
    assignments = [
        :( dest.$name = src.$name ) for name in fieldnames(MT)
    ]
    quote $(assignments...) end
end

function copyto!(dest::MT, src::TT) where {MT,TT}
    for name in (fieldnames(MT))
        if !isa(src[name], ChainRulesCore.ZeroTangent) && !isa(getfield(dest,name), Int)
            setfield!(dest, name, convert(typeof(getfield(dest, name)), src[name]))
        end
    end
end

to_named_tuple(p) = (; (v=>getfield(p, v) for v in fieldnames(typeof(p)))...)

function create_tangent(shadowmodel::MT) where {MT}
    shadowtuple = to_named_tuple(shadowmodel)
    return Tangent{MT,typeof(shadowtuple)}(shadowtuple)
end

function set_zero!(::Ptr{Nothing})
    return nothing
end

function set_zero!(nestedmodel::MT) where {MT}
    if length(fieldnames(MT)) == 0
        if isreal(nestedmodel)
            if isa(nestedmodel, Number)
                nestedmodel = zero(MT)
            else
                fill!(nestedmodel, zero(eltype(nestedmodel)))
            end
        end
    else
        for name in fieldnames(MT)
            set_zero!(getfield(nestedmodel, name))
        end
    end
end

function ChainRulesCore.rrule(
    ::typeof(Checkpointing.checkpoint_struct_for),
    body::Function,
    alg::Scheme,
    model::MT,
    shadowmodel::MT,
    range::Function
) where {MT}
    model_input = deepcopy(model)
    for i in 1:alg.steps
        body(model)
    end
    function checkpoint_struct_pullback(dmodel)
        set_zero!(shadowmodel)
        copyto!(shadowmodel, dmodel)
        model = checkpoint_struct_for(body, alg, model_input, shadowmodel, range)
        dshadowmodel = create_tangent(shadowmodel)
        return NoTangent(), NoTangent(), NoTangent(), dshadowmodel, NoTangent(), NoTangent()
    end
    return model, checkpoint_struct_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Checkpointing.checkpoint_struct_while),
    body::Function,
    alg::Scheme,
    model::MT,
    shadowmodel::MT,
    condition::Function
) where {MT}
    model_input = deepcopy(model)
    while condition(model)
        body(model)
    end
    function checkpoint_struct_pullback(dmodel)
        set_zero!(shadowmodel)
        copyto!(shadowmodel, dmodel)
        model = checkpoint_struct_while(body, alg, model_input, shadowmodel, condition)
        dshadowmodel = create_tangent(shadowmodel)
        return NoTangent(), NoTangent(), NoTangent(), dshadowmodel, NoTangent(), NoTangent()
    end
    return model, checkpoint_struct_pullback
end

"""
    @checkpoint_struct(
        alg,
        model,
        loop,
    )

This macro is supposed to be only used in conjunction with ChainRules. It does
not initialize the shadowcopy. Apply the checkpointing scheme `alg` on the loop
`loop` expression. `model` is the primal struct. `shadowmodel` contains the
adjoints and is created here.  It is supposed to be initialized by ChainRules.

"""
macro checkpoint_struct(alg, model, loop)
    if loop.head == :for
        ex = quote
            shadowmodel = deepcopy($model)
            function range()
                $(loop.args[1])
            end
            $model = checkpoint_struct_for($alg, $model, shadowmodel, range) do $model
                $(loop.args[2])
            end
        end
    elseif loop.head == :while
        ex = quote
            shadowmodel = deepcopy($model)
            function condition($model)
                $(loop.args[1])
            end
            $model = checkpoint_struct_while($alg, $model, shadowmodel, condition) do $model
                $(loop.args[2])
                nothing
            end
        end
    else
        error("Checkpointing.jl: Unknown loop construct.")
    end
    esc(ex)
end

"""
    @checkpoint_struct(
        alg,
        model,
        shadowmodel,
        loop,
    )

Apply the checkpointing scheme `alg` on the loop `loop` expression. `model` is
the primal struct and `shadowmodel` the adjoint struct where the adjoints
are seeded and retrieved.

"""
macro checkpoint_struct(alg, model, shadowmodel, loop)
    ex = quote
        function range()
            $(loop.args[1])
        end
        $model = checkpoint_struct_for($alg, $model, $shadowmodel, range) do $model
            $(loop.args[2])
            nothing
        end
    end
    esc(ex)
end

"""
    checkpoint_struct(
        body::Function
        alg::Scheme,
        model::MT,
        shadowmodel::MT,
    ) where {MT}

Default method for the function `checkpoint_struct` if the function is not specialized for an unknown scheme `Scheme`.
`body` is the loop body as generated by the macro `@checkpoint_struct` and `MT` is the checkpointed struct.

"""
function checkpoint_struct(
        body::Function,
        alg::Scheme,
        model_input::MT,
        shadowmodel::MT
    ) where {MT}
    error("No checkpointing scheme implemented for algorithm $Scheme.")
end

end
