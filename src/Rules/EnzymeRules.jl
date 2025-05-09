using Enzyme
import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules
import EnzymeCore

function augmented_primal(
    config,
    func::Const{typeof(Checkpointing.checkpoint_struct_for)},
    ret,
    body,
    alg,
    range,
)
    # tape_model = deepcopy(model.val)
    @show typeof(alg)
    tape_body = deepcopy(body.val)
    for fieldname in fieldnames(typeof(body.val))
        field = getfield(body.val, fieldname)
        if isa(field, Core.Box)
            error("[Checkpointing.jl]: Variable $fieldname is reassigned inside the loop. Please make sure that $fieldname is only changed inplace")
        end
    end
    make_zero!(body.dval)
    func.val(body.val, alg.val, range.val)
    if needs_primal(config)
        return AugmentedReturn(nothing, nothing, (tape_body,))
    else
        return AugmentedReturn(nothing, nothing, (tape_body,))
    end
end

function reverse(
    config,
    ::Const{typeof(Checkpointing.checkpoint_struct_for)},
    dret::Type{<:Const},
    tape,
    body::Union{Duplicated, MixedDuplicated},
    alg,
    range,
)
    (body_input,) = tape
    @show typeof(body.dval)
    @show typeof(body)
    dbody = if isa(body, Duplicated)
        dbody = body.dval
    elseif isa(body, MixedDuplicated)
        dbody = body.dval[]
    else
        error("Checkpointing.jl: Unknown type of dbody")
    end

    Checkpointing.rev_checkpoint_struct_for(
        config,
        body_input,
        dbody,
        alg.val,
        range.val,
    )
    return (nothing, nothing, nothing)
end

function augmented_primal(
    config,
    func::Const{typeof(Checkpointing.checkpoint_struct_while)},
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
    func.val(body.val, alg.val)
    if needs_primal(config)
        return AugmentedReturn(nothing, nothing, (tape_body,))
    else
        return AugmentedReturn(nothing, nothing, (tape_body,))
    end
end

function reverse(
    config,
    ::Const{typeof(Checkpointing.checkpoint_struct_while)},
    dret::Type{<:Const},
    tape,
    body::Union{Duplicated, MixedDuplicated},
    alg,
)
    (body_input,) = tape
    dbody = if isa(body, Duplicated)
        dbody = body.dval
    elseif isa(body, MixedDuplicated)
        dbody = body.dval[]
    else
        error("Checkpointing.jl: Unknown type of dbody")
    end

    Checkpointing.rev_checkpoint_struct_while(
        config,
        body_input,
        dbody,
        alg.val,
    )
    return (nothing, nothing)
end

export augmented_primal, reverse
