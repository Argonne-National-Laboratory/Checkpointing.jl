module Checkpointing

import CheckpointingCore
# One @ad_checkpoint: CheckpointingCore marks loops with the compiler's
# schedules, and runs any other scheme through checkpoint_for and
# checkpoint_while, whose methods for the schemes here are below.
import CheckpointingCore: @ad_checkpoint, checkpoint_for, checkpoint_while

using Serialization
import EnzymeCore

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
export enzyme_scheme, EnzymeLLVM

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
    _new_storage(S, checkpoints)

Build the placeholder storage for a scheme that does not yet know its loop body.

`S` is the storage *type*, not its name. Taking a `Symbol` and `eval`ing it -- as
this used to -- cannot be precompiled, bumps the world age at the call site, and
cannot name a type that lives in a package extension or in the caller's own
module.
"""
_new_storage(S::Type, checkpoints::Integer) = S{Nothing}(checkpoints)

_new_storage(S::Symbol, ::Integer) = throw(
    ArgumentError(
        "[Checkpointing.jl]: `storage` takes the storage type itself, not its name. " *
        "Use `storage = $S` instead of `storage = :$S`.",
    ),
)

include("Storage/copy.jl")
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
include("EnzymeABI.jl")

function __init__()
    _init_enzyme_abi()
end

"""
    enzyme_marks_loops()

Whether the loaded Enzyme.jl checkpoints a loop marked with
CheckpointingCore's loop annotation itself.
"""
enzyme_marks_loops() = isdefined(Enzyme.Compiler, :keep_checkpoint_loops!)

end
