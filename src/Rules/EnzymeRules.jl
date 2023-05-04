using Enzyme
# import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules

function augmented_primal(
    config,
    func::Const{typeof(Checkpointing.checkpoint_struct_for)},
    ret,
    body,
    alg,
    model,
    range,
)
    if needs_primal(config)
        return AugmentedReturn(func.val(body.val, alg.val, model.val, range.val), nothing, (model.val,))
    else
        return AugmentedReturn(nothing, nothing, (model.val,))
    end
end

function reverse(
    config::ConfigWidth{1},
    ::Const{typeof(Checkpointing.checkpoint_struct_for)},
    dret::Type{<:Const},
    tape,
    body,
    alg,
    model::Duplicated,
    range,
)
    (model_input,) = tape
    model_final = Checkpointing.rev_checkpoint_struct_for(
        body.val,
        alg.val,
        model_input,
        model.dval,
        range.val
    )
    copyto!(model.val, model_final)
    return (nothing, nothing, nothing, nothing)
end

function augmented_primal(
    config,
    func::Const{typeof(Checkpointing.checkpoint_struct_while)},
    ret,
    body,
    alg,
    model,
    condition,
)
    println("augmented_primal")
    if needs_primal(config)
        return AugmentedReturn(func.val(body.val, alg.val, model.val, condition.val), nothing, (model.val,))
    else
        return AugmentedReturn(nothing, nothing, (model.val,))
    end
end

function reverse(
    config::ConfigWidth{1},
    ::Const{typeof(Checkpointing.checkpoint_struct_while)},
    dret::Type{<:Const},
    tape,
    body,
    alg,
    model::Duplicated,
    condition,
)
    (model_input,) = tape
    model_final = Checkpointing.rev_checkpoint_struct_while(
        body.val,
        alg.val,
        model_input,
        model.dval,
        condition.val
    )
    copyto!(model.val, model_final)
    return (nothing, nothing, nothing, nothing)
end

export augmented_primal, reverse
