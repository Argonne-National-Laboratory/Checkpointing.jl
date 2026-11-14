using Enzyme
import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules
import EnzymeCore

# Floating-point scalar types that can cause activity analysis issues when captured
# in closures alongside mutable structs. Integer types are always inactive in AD,
# so they don't cause the same problem.
const PROBLEMATIC_SCALAR_TYPES = Union{Float16,Float32,Float64}

"""
    check_closure_captures(body)

Check the closure for common issues that cause Enzyme activity analysis errors.
Provides helpful error messages pointing to the specific problematic variables.
"""
function check_closure_captures(body)
    closure_type = typeof(body.val)
    field_names = fieldnames(closure_type)
    field_types = fieldtypes(closure_type)

    # Find captured scalar variables and struct variables
    scalar_vars = String[]
    struct_vars = String[]

    for (name, ftype) in zip(field_names, field_types)
        field = getfield(body.val, name)
        if isa(field, Core.Box)
            error(
                "[Checkpointing.jl]: Variable `$name` is reassigned inside the loop. " *
                "Please make sure that `$name` is only modified in-place.",
            )
        elseif ftype <: PROBLEMATIC_SCALAR_TYPES
            push!(scalar_vars, string(name))
        elseif ismutabletype(ftype)
            push!(struct_vars, string(name))
        end
    end

    # If we have both floating-point scalars and mutable structs captured, warn the user.
    # This combination causes Enzyme activity analysis errors because Enzyme can't
    # determine whether the floating-point values should participate in AD.
    return if !isempty(scalar_vars) && !isempty(struct_vars)
        scalar_list = join(["`$v`" for v in scalar_vars], ", ")
        struct_list = join(["`$v`" for v in struct_vars], ", ")
        error(
            "[Checkpointing.jl]: The loop body captures floating-point variable(s) $scalar_list " *
            "alongside mutable struct(s) $struct_list.\n" *
            "This causes Enzyme activity analysis errors.\n\n" *
            "Solution: Store these values as fields in your mutable struct instead of " *
            "capturing them as separate variables.\n\n" *
            "Example - instead of:\n" *
            "    h = 0.1\n" *
            "    @ad_checkpoint scheme for i in 1:n\n" *
            "        model.t += h  # ERROR: h captured from outer scope\n" *
            "    end\n\n" *
            "Use:\n" *
            "    model.h = 0.1  # store in struct\n" *
            "    @ad_checkpoint scheme for i in 1:n\n" *
            "        model.t += model.h  # OK: access through struct\n" *
            "    end",
        )
    end
end


"""
    shadow(body)

The shadow (adjoint) closure paired with an annotated loop body.

Enzyme hands the loop body over as a `Duplicated`, `MixedDuplicated` or `Const`
annotation, or in vector mode (`EnzymeRules.width(config) > 1`) as a
`BatchDuplicated` or `BatchMixedDuplicated`, whose shadow is a tuple with one
closure per lane. Picking the shadow apart with a chain of `isa` tests infers as
`Any`, which puts a dynamic dispatch at the entry to the whole reverse sweep and
rebuilds `Duplicated(body, dbody)` from an `Any` on every iteration of the hot
loop. Dispatching instead keeps the shadow's type concrete.
"""
shadow(body::Duplicated) = body.dval
shadow(body::MixedDuplicated) = body.dval[]
shadow(body::BatchDuplicated) = body.dval
shadow(body::BatchMixedDuplicated) = map(getindex, body.dval)

function shadow(body::Const)
    # Mixed activity (a closure capturing both a mutable struct and scalars)
    # makes Enzyme mark the body `Const`. `check_closure_captures` usually
    # pinpoints the offending capture; fall back to a generic message.
    check_closure_captures(body)
    return error(
        "[Checkpointing.jl]: The loop body was marked as Const by Enzyme, " *
        "but checkpointing requires an active (Duplicated) closure. " *
        "Make sure your loop body captures a mutable struct that is being differentiated.",
    )
end

shadow(body) = error("Checkpointing.jl: Unknown annotation type for body: $(typeof(body))")

"""
    adjoint_step!(config, body, dbody, args...)

Differentiate one loop iteration `body(args...)` at the current primal state of
`body`, accumulating its adjoint into the shadow closure `dbody`.

In vector mode `dbody` is a tuple with one shadow closure per lane. The
schedule, the checkpoints and the recomputation only touch the primal `body`,
so a batched reverse sweep differs from a scalar one only here: each reversed
iteration is differentiated once for all lanes. Enzyme takes the batch width
from the arguments, not from the annotation of the function itself, so the
batched body is passed as an argument to a `Const` trampoline.
"""
function adjoint_step!(config, body, dbody, args...)
    Enzyme.autodiff(
        EnzymeCore.set_runtime_activity(Reverse, config),
        Duplicated(body, dbody),
        Const,
        args...,
    )
    return nothing
end

function adjoint_step!(config, body, dbody::NTuple{N,Any}, args...) where {N}
    Enzyme.autodiff(
        EnzymeCore.set_runtime_activity(Reverse, config),
        Const(_call),
        Const,
        BatchDuplicated(body, dbody),
        args...,
    )
    return nothing
end

_call(body, args...) = body(args...)

"""
    fwd_checkpoint_for(config, body::Annotation, alg, range) -> tape

The forward half of a scheme as the rule calls it, with the Enzyme `config` and
the annotated loop body. Most schemes only run and store the primal in their
forward half and implement `fwd_checkpoint_for(body, alg, range)`, which this
falls back to. A scheme that differentiates in the forward half already, as
[`CAMS`](@ref) does to keep tapes, extends this method instead and gets the
shadow from [`shadow`](@ref).
"""
fwd_checkpoint_for(config, body::Annotation, alg::Scheme, range) =
    fwd_checkpoint_for(body.val, alg, range)

# --- split mode ------------------------------------------------------------

"""
    split_thunks(config, body, dbody, I) -> (forward, reverse, Tape)

Enzyme's split reverse mode for one loop iteration `body(::I)`: `forward` runs
the iteration and returns a tape, from which `reverse` later accumulates the
adjoint of that iteration into `dbody` without running it again. Used through
[`tape_step!`](@ref) and [`adjoint_tape!`](@ref).

The tape holds every value the reverse pass reads, so the state of `body` may
change freely between the two calls.
"""
function split_thunks(config, body::FT, dbody, ::Type{I}) where {FT,I}
    mode = EnzymeCore.set_runtime_activity(ReverseSplitNoPrimal, config)
    forward, reverse = autodiff_thunk(mode, Duplicated{FT}, Const, Const{I})
    return forward, reverse, Enzyme.tape_type(mode, Duplicated{FT}, Const, Const{I})
end

function split_thunks(config, body::FT, dbody::NTuple{N,Any}, ::Type{I}) where {FT,N,I}
    mode = ReverseSplitWidth(
        EnzymeCore.set_runtime_activity(ReverseSplitNoPrimal, config),
        Val(N),
    )
    FA = Const{typeof(_call)}
    BT = BatchDuplicated{FT,N}
    forward, reverse = autodiff_thunk(mode, FA, Const, BT, Const{I})
    return forward, reverse, Enzyme.tape_type(mode, FA, Const, BT, Const{I})
end

"""
    tape_step!(forward, body, dbody, i) -> tape

Run iteration `body(i)` forward in split mode and return its tape.
"""
tape_step!(forward, body, dbody, i) = forward(Duplicated(body, dbody), Const(i))[1]
tape_step!(forward, body, dbody::NTuple{N,Any}, i) where {N} =
    forward(Const(_call), BatchDuplicated(body, dbody), Const(i))[1]

"""
    adjoint_tape!(reverse, body, dbody, i, tape)

Accumulate the adjoint of iteration `body(i)` into `dbody` from the `tape` that
[`tape_step!`](@ref) returned for it. A tape can be used once.
"""
function adjoint_tape!(reverse, body, dbody, i, tape)
    reverse(Duplicated(body, dbody), Const(i), tape)
    return nothing
end

function adjoint_tape!(reverse, body, dbody::NTuple{N,Any}, i, tape) where {N}
    reverse(Const(_call), BatchDuplicated(body, dbody), Const(i), tape)
    return nothing
end

# --- CAMS ------------------------------------------------------------------

function fwd_checkpoint_for(config, body::Annotation, alg::CAMS, range)
    @assert alg.steps == length(range)
    alg.steps == 0 && return nothing
    dbody = shadow(body)
    forward, reverse, Tape = split_thunks(config, body.val, dbody, eltype(range))
    ntapes = 0
    for act in alg.actions
        act.kind == cams_tape && (ntapes = max(ntapes, act.b))
    end
    tapes = Vector{Tape}(undef, ntapes)
    return _cams_forward(forward, reverse, tapes, body.val, dbody, alg, range)
end

# Behind a function barrier, so that the thunks and the tapes are concretely
# typed in the sweep.
function _cams_forward(forward, reverse, tapes, body, dbody, alg::CAMS, range)
    actions = alg.actions
    n = 1
    while true
        act = actions[n]
        if act.kind == cams_advance
            for j = act.a:act.b
                body(range[j])
            end
        elseif act.kind == cams_store
            save!(alg.storage, body, act.a)
        elseif act.kind == cams_tape
            tapes[act.b] = tape_step!(forward, body, dbody, range[act.a])
        elseif act.kind == cams_adjoint
            # The first step to be adjoined is the last one of the loop. It is
            # run in split mode, which leaves `body` in the final state, and
            # its tape is handed to the reverse pass.
            @assert act.a == alg.steps
            last = tape_step!(forward, body, dbody, range[act.a])
            # The reverse sweep recomputes on `body` itself, which the tapes
            # are tied to, and puts the final state back when it is done.
            return (forward, reverse, tapes, last, checkpoint_alloc(body), body, n)
        else
            error("CAMS: unexpected action $(act.kind) before the first adjoint")
        end
        n += 1
    end
end

function rev_checkpoint_for(config, tape, dbody, alg::CAMS, range)
    tape === nothing && return nothing
    forward, reverse, tapes, last, final, body, n = tape
    _cams_reverse(config, forward, reverse, tapes, last, body, dbody, alg, range, n)
    checkpoint_copy!(body, final)
    return nothing
end

function _cams_reverse(config, forward, reverse, tapes, last, body, dbody, alg, range, n)
    actions = alg.actions
    adjoint_tape!(reverse, body, dbody, range[actions[n].a], last)
    for m = (n+1):length(actions)
        act = actions[m]
        if act.kind == cams_advance
            for j = act.a:act.b
                body(range[j])
            end
        elseif act.kind == cams_store
            save!(alg.storage, body, act.a)
        elseif act.kind == cams_restore
            load!(body, alg.storage, act.a)
        elseif act.kind == cams_tape
            tapes[act.b] = tape_step!(forward, body, dbody, range[act.a])
        elseif act.kind == cams_adjoint
            adjoint_step!(config, body, dbody, Const(range[act.a]))
        else
            adjoint_tape!(reverse, body, dbody, range[act.a], tapes[act.b])
        end
    end
    return nothing
end

function augmented_primal(
    config,
    func::Const{typeof(Checkpointing.checkpoint_for)},
    ret,
    body,
    alg,
    range,
)
    check_closure_captures(body)
    # Drive the scheme's schedule in the primal instead of a plain loop. The
    # schedule up to the first u-turn *is* a forward sweep, with stores in it;
    # running a plain loop here and replaying that sweep from the initial state
    # in `reverse` did the whole forward pass twice.
    scheme = instantiate(typeof(body.val), alg.val, length(range.val))
    fwd_tape = Checkpointing.fwd_checkpoint_for(config, body, scheme, range.val)
    return AugmentedReturn(nothing, nothing, (scheme, fwd_tape))
end

function reverse(
    config,
    ::Const{typeof(Checkpointing.checkpoint_for)},
    dret::Type{<:Const},
    tape,
    body::Union{Const,Duplicated,MixedDuplicated,BatchDuplicated,BatchMixedDuplicated},
    alg,
    range,
)
    scheme, fwd_tape = tape
    dbody = shadow(body)

    Checkpointing.rev_checkpoint_for(config, fwd_tape, dbody, scheme, range.val)
    return (nothing, nothing, nothing)
end

function augmented_primal(
    config,
    func::Const{typeof(Checkpointing.checkpoint_while)},
    ret,
    body,
    alg,
)
    check_closure_captures(body)
    scheme = instantiate(typeof(body.val), alg.val)
    fwd_tape = Checkpointing.fwd_checkpoint_while(body.val, scheme)
    return AugmentedReturn(nothing, nothing, (scheme, fwd_tape))
end

function reverse(
    config,
    ::Const{typeof(Checkpointing.checkpoint_while)},
    dret::Type{<:Const},
    tape,
    body::Union{Const,Duplicated,MixedDuplicated,BatchDuplicated,BatchMixedDuplicated},
    alg,
)
    scheme, fwd_tape = tape
    dbody = shadow(body)

    Checkpointing.rev_checkpoint_while(config, fwd_tape, dbody, scheme)
    return (nothing, nothing)
end
