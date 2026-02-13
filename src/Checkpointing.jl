module Checkpointing

using LinearAlgebra
using DataStructures
using Serialization
using HDF5
using Adapt
using KernelAbstractions: get_backend, GPU

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
export @ad_checkpoint, checkpoint_for, checkpoint_while
export instantiate
export reset!
export AbstractStorage, ArrayStorage, HDF5Storage
export Revolve, Periodic, Online_r2
export serialize, deserialize

function serialize(x)
    s = IOBuffer()
    Serialization.serialize(s, x)
    take!(s)
end

function deserialize(x)
    s = IOBuffer(x)
    Serialization.deserialize(s)
end


abstract type AbstractStorage end

"""
    is_gpu_array(x) -> Bool

Return `true` if `x` is a GPU array (CuArray, ROCArray, oneArray, etc.).
Uses KernelAbstractions.jl's `get_backend` to detect GPU arrays.
"""
is_gpu_array(x::AbstractArray) = get_backend(x) isa GPU
is_gpu_array(::Any) = false

"""
    _check_gpu_fields(x)

Recursively check whether `x` or any of its fields contain GPU arrays.
Throws an `ArgumentError` if a GPU array is found.
"""
function _check_gpu_fields(x)
    is_gpu_array(x) && throw(
        ArgumentError(
            "GPU arrays are not supported with HDF5Storage or ChkpDump serialization. " *
            "Use ArrayStorage instead.",
        ),
    )
    return if ismutable(x) && !isa(x, AbstractArray)
        for name in fieldnames(typeof(x))
            _check_gpu_fields(getfield(x, name))
        end
    end
end

"""
    check_no_gpu_arrays(body::Function)

Walk the captured fields of a closure to detect GPU arrays. Throws an
`ArgumentError` if any are found.
"""
function check_no_gpu_arrays(body::Function)
    for name in fieldnames(typeof(body))
        _check_gpu_fields(getfield(body, name))
    end
    return
end

include("Storage/ArrayStorage.jl")
include("Storage/HDF5Storage.jl")
include("ChkpDump.jl")


include("Schemes/Revolve.jl")
include("Schemes/Periodic.jl")
include("Schemes/Online_r2.jl")

function checkpoint_for(body::Function, scheme::Scheme, range)
    for i in range
        body(i)
    end
    return nothing
end

function checkpoint_while(body::Function, scheme::Scheme)
    go = true
    while go
        go = body()
    end
    return nothing
end

include("Rules/EnzymeRules.jl")

"""
    @ad_checkpoint(
        alg,
        loop,
    )

This macro is supposed to be only used in conjunction with EnzymeRules. It does
not initialize the shadowcopy. Apply the checkpointing scheme `alg` on the loop
`loop` expression.
"""
macro ad_checkpoint(alg, loop)
    range = loop.args[1].args[2]
    _iterator = loop.args[1].args[1]
    body = loop.args[2]
    i = gensym()
    fbody = gensym("fbody")
    wbody = gensym("wbody")
    if loop.head == :for
        range = loop.args[1].args[2]
        ex = quote
            let
                if !isa($range, UnitRange{Int64})
                    error(
                        "Checkpointing.jl: Only UnitRange{Int64} is supported. range = $(typeof($range)) is not supported.",
                    )
                end
                $fbody = $i -> begin
                    $_iterator = $i
                    $body
                end
                Checkpointing.checkpoint_for($fbody, $alg, $range)
            end
        end
    elseif loop.head == :while
        ex = quote
            $wbody = () -> begin
                $body
                # return loop condition
                return $(loop.args[1])
            end
            Checkpointing.checkpoint_while($wbody, $alg)
        end
    else
        error("Checkpointing.jl: Unknown loop construct.")
    end
    esc(ex)
end

end
