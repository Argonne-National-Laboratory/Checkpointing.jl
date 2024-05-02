using Enzyme
import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
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
    primal = func.val(body.val, alg.val, deepcopy(model.val), range.val)
    if needs_primal(config)
        return AugmentedReturn(primal, nothing, (model.val,))
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
        range.val,
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
    primal = func.val(body.val, alg.val, deepcopy(model.val), condition.val)
    if needs_primal(config)
        return AugmentedReturn(primal, nothing, (model.val,))
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
        condition.val,
    )
    copyto!(model.val, model_final)
    return (nothing, nothing, nothing, nothing)
end

export augmented_primal, reverse
