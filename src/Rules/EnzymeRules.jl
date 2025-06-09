using Enzyme
import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules
import EnzymeCore

function augmented_primal(
    config,
    func::Const{typeof(Checkpointing.checkpoint_for)},
    ret,
    body,
    alg,
    range,
)
    tape_body = deepcopy(body.val)
    for fieldname in fieldnames(typeof(body.val))
        field = getfield(body.val, fieldname)
        if isa(field, Core.Box)
            error("[Checkpointing.jl]: Variable $fieldname is reassigned inside the loop. Please make sure that $fieldname is only changed inplace")
        end
    end
    # make_zero!(body.dval)
    func.val(body.val, alg.val, range.val)
    if needs_primal(config)
        return AugmentedReturn(nothing, nothing, (tape_body,))
    else
        return AugmentedReturn(nothing, nothing, (tape_body,))
    end
end

function reverse(
    config,
    ::Const{typeof(Checkpointing.checkpoint_for)},
    dret::Type{<:Const},
    tape,
    body::Union{Duplicated, MixedDuplicated},
    alg,
    range,
)
    (body_input,) = tape
    scheme = instantiate(typeof(body_input), alg.val, length(range.val))
    dbody = if isa(body, Duplicated)
        dbody = body.dval
    elseif isa(body, MixedDuplicated)
        dbody = body.dval[]
    else
        error("Checkpointing.jl: Unknown type of dbody")
    end

    Checkpointing.rev_checkpoint_for(
        config,
        body_input,
        dbody,
        scheme,
        range.val,
    )
    return (nothing, nothing, nothing)
end

function augmented_primal(
    config,
    func::Const{typeof(Checkpointing.checkpoint_while)},
    ret,
    body,
    alg,
)
    tape_body = deepcopy(body.val)
    for fieldname in fieldnames(typeof(body.val))
        field = getfield(body.val, fieldname)
        if isa(field, Core.Box)
            error("[Checkpointing.jl]: Variable $fieldname is reassigned inside the loop. Please make sure that $fieldname is only changed inplace")
        end
    end
    # make_zero!(body.dval)
    func.val(body.val, alg.val)
    if needs_primal(config)
        return AugmentedReturn(nothing, nothing, (tape_body,))
    else
        return AugmentedReturn(nothing, nothing, (tape_body,))
    end
end

function reverse(
    config,
    ::Const{typeof(Checkpointing.checkpoint_while)},
    dret::Type{<:Const},
    tape,
    body::Union{Duplicated, MixedDuplicated},
    alg,
)
    (body_input,) = tape
    scheme = instantiate(typeof(body_input), alg.val)
    dbody = if isa(body, Duplicated)
        dbody = body.dval
    elseif isa(body, MixedDuplicated)
        dbody = body.dval[]
    else
        error("Checkpointing.jl: Unknown type of dbody")
    end

    Checkpointing.rev_checkpoint_while(
        config,
        body_input,
        dbody,
        scheme,
    )
    return (nothing, nothing)
end

export augmented_primal, reverse
