module Checkpointing

using LinearAlgebra
using DataStructures
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

export Scheme
export @checkpoint, @checkpoint_struct, checkpoint_struct_for, checkpoint_struct_while
export reset!

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
include("ChkpDump.jl")

export AbstractStorage, ArrayStorage, HDF5Storage

include("Schemes/Revolve.jl")
include("Schemes/Periodic.jl")
include("Schemes/Online_r2.jl")

export Revolve, guess, factor, next_action!, ActionFlag, Periodic
export Online_r2, update_revolve

@generated function copyto!(dest::MT, src::MT) where {MT}
    assignments = [:(dest.$name = src.$name) for name in fieldnames(MT)]
    quote
        $(assignments...)
    end
end

function copyto!(dest::MT, src::TT) where {MT,TT}
    for name in (fieldnames(MT))
        if !isa(src[name], ChainRulesCore.ZeroTangent) && !isa(getfield(dest, name), Int)
            setfield!(dest, name, convert(typeof(getfield(dest, name)), src[name]))
        end
    end
end

to_named_tuple(p) = (; (v => getfield(p, v) for v in fieldnames(typeof(p)))...)

function create_tangent(shadowmodel::MT) where {MT}
    shadowtuple = to_named_tuple(shadowmodel)
    return Tangent{MT,typeof(shadowtuple)}(shadowtuple)
end

function set_zero!(::Ptr{Nothing})
    return nothing
end

function set_zero!(nestedmodel::MT) where {MT}
    if length(fieldnames(MT)) == 0
        if eltype(nestedmodel) <: Number && isreal(nestedmodel)
            if isa(nestedmodel, Number)
                nestedmodel = zero(MT)
            else
                fill!(nestedmodel, zero(eltype(nestedmodel)))
            end
        end
    else
        for name in fieldnames(MT)
            field = getfield(nestedmodel, name)
            if (!isa(field, DataType) && !isa(field, Symbol) && !isa(field, String))
                set_zero!(getfield(nestedmodel, name))
            end
        end
    end
end

function checkpoint_struct_for(body::Function, scheme::Scheme, model, range)
    for gensym() in range
        body(model)
    end
    return model
end

function checkpoint_struct_while(body::Function, scheme::Scheme, model, condition)
    while condition(model)
        body(model)
    end
    return model
end

include("Rules/ChainRules.jl")
include("Rules/EnzymeRules.jl")

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
        body = loop.args[2]
        iterator = loop.args[1].args[1]
        from = loop.args[1].args[2].args[2]
        to = loop.args[1].args[2].args[3]
        range = loop.args[1].args[2]
        ex = quote
            let
                if !isa($range, UnitRange{Int64})
                    error("Checkpointing.jl: Only UnitRange{Int64} is supported.")
                end
                $iterator = $from
                $model = Checkpointing.checkpoint_struct_for(
                    $alg,
                    $model,
                    $(loop.args[1].args[2]),
                ) do $model
                    $body
                    $iterator += 1
                    nothing
                end
            end
        end
    elseif loop.head == :while
        ex = quote
            function condition($model)
                $(loop.args[1])
            end
            $model =
                Checkpointing.checkpoint_struct_while($alg, $model, condition) do $model
                    $(loop.args[2])
                    nothing
                end
        end
    else
        error("Checkpointing.jl: Unknown loop construct.")
    end
    esc(ex)
end

function fwd_checkpoint_struct_for(
    body::Function,
    scheme::Scheme,
    model,
    range::UnitRange{Int64},
)
    for i in range
        body(model)
    end
    return model
end

function fwd_checkpoint_struct_while(body::Function, scheme::Scheme, model, condition)
    while condition(model)
        body(model)
    end
    return model
end
end
