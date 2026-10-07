module Checkpointing

import CheckpointingCore

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

"""
    @ad_checkpoint(
        alg,
        loop,
    )

Apply the checkpointing scheme `alg` on the loop `loop` expression.

A `for` loop with `Revolve(k)` or `Periodic(k)` and a literal `k`, when the
loaded Enzyme.jl checkpoints marked loops, stays a plain loop marked with
CheckpointingCore's loop annotation: Enzyme reverses it with that schedule,
and the loop body needs no closure. Otherwise the loop body becomes a
closure that `checkpoint_for` or `checkpoint_while` runs, which the
EnzymeRules reverse with `alg`; it does not initialize the shadow copy.
"""
macro ad_checkpoint(alg, loop)
    if loop isa Expr && loop.head === :for && enzyme_marks_loops()
        s = CheckpointingCore.checkpoint_schedule(alg)
        if s !== nothing && s[1] in (:revolve, :periodic)
            return esc(CheckpointingCore.annotated_loop(s..., loop))
        end
    end
    body = loop.args[2]
    i = gensym()
    fbody = gensym("fbody")
    wbody = gensym("wbody")
    rng = gensym("range")
    if loop.head == :for
        # Only reach into the loop header once the loop kind is known: a
        # `while` condition may be a bare symbol, which has no `.args`.
        _iterator = loop.args[1].args[1]
        range = loop.args[1].args[2]
        ex = quote
            let
                # Bind the range once -- interpolating `$range` at each use
                # would evaluate the user's expression several times.
                $rng = $range
                if !isa($rng, UnitRange{Int64})
                    error(
                        "Checkpointing.jl: Only UnitRange{Int64} is supported. range = $(typeof($rng)) is not supported.",
                    )
                end
                $fbody = $i -> begin
                    $_iterator = $i
                    $body
                end
                Checkpointing.checkpoint_for($fbody, $alg, $rng)
            end
        end
    elseif loop.head == :while
        ex = quote
            let
                $wbody = () -> begin
                    $body
                    # return loop condition
                    return $(loop.args[1])
                end
                Checkpointing.checkpoint_while($wbody, $alg)
            end
        end
    else
        error("Checkpointing.jl: Unknown loop construct.")
    end
    esc(ex)
end

end
