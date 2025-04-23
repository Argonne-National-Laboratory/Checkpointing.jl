using Enzyme
import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules
import EnzymeCore

function maybe_duplicated(f, df)
    if !Enzyme.Compiler.guaranteed_const(typeof(f))
        return DuplicatedNoNeed(f, df)
    else
        return Const(f)
    end
end

function augmented_primal(
    config,
    func::Const{typeof(Checkpointing.checkpoint_struct_for)},
    ret,
    body,
    alg,
    model,
    range,
)
    tape_model = deepcopy(model.val)
    func.val(body.val, alg.val, model.val, range.val)
    if needs_primal(config)
        return AugmentedReturn(nothing, nothing, (tape_model,))
    else
        return AugmentedReturn(nothing, nothing, (tape_model,))
    end
end

function reverse(
    config,
    ::Const{typeof(Checkpointing.checkpoint_struct_for)},
    dret::Type{<:Const},
    tape,
    body,
    alg,
    model::Duplicated,
    range,
)
    (model_input,) = tape
    Checkpointing.rev_checkpoint_struct_for(
        config,
        body.val,
        alg.val,
        model_input,
        model.dval,
        range.val,
    )
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
    tape_model = deepcopy(model.val)
    func.val(body.val, alg.val, model.val, condition.val)
    if needs_primal(config)
        return AugmentedReturn(nothing, nothing, (tape_model,))
    else
        return AugmentedReturn(nothing, nothing, (tape_model,))
    end
end

function reverse(
    config,
    ::Const{typeof(Checkpointing.checkpoint_struct_while)},
    dret::Type{<:Const},
    tape,
    body,
    alg,
    model::Duplicated,
    condition,
)
    (model_input,) = tape
    Checkpointing.rev_checkpoint_struct_while(
        config,
        body.val,
        alg.val,
        model_input,
        model.dval,
        condition.val,
    )
    return (nothing, nothing, nothing, nothing)
end

export augmented_primal, reverse
