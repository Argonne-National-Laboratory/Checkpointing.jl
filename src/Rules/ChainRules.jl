using ChainRulesCore

function ChainRulesCore.rrule(
    ::typeof(Checkpointing.checkpoint_struct_for),
    body::Function,
    alg::Scheme,
    model::MT,
    range,
) where {MT}
    # Augmented forward run
    # TODO: store checkpoints during this forward call and
    # start the reverse with first uturn
    model_input = deepcopy(model)
    model = fwd_checkpoint_struct_for(
        body,
        alg,
        model,
        range,
    )
    function checkpoint_struct_pullback(dmodel)
        shadowmodel = deepcopy(model_input)
        set_zero!(shadowmodel)
        copyto!(shadowmodel, dmodel)
        model = rev_checkpoint_struct_for(body, alg, model_input, shadowmodel, range)
        dshadowmodel = create_tangent(shadowmodel)
        return NoTangent(), NoTangent(), NoTangent(), dshadowmodel, NoTangent(), NoTangent()
    end
    return model, checkpoint_struct_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Checkpointing.checkpoint_struct_while),
    body::Function,
    alg::Scheme,
    model::MT,
    condition::Function
) where {MT}
    model_input = deepcopy(model)
    while condition(model)
        body(model)
    end
    function checkpoint_struct_pullback(dmodel)
        shadowmodel = deepcopy(model_input)
        set_zero!(shadowmodel)
        copyto!(shadowmodel, dmodel)
        model = rev_checkpoint_struct_while(body, alg, model_input, shadowmodel, condition)
        dshadowmodel = create_tangent(shadowmodel)
        return NoTangent(), NoTangent(), NoTangent(), dshadowmodel, NoTangent(), NoTangent()
    end
    return model, checkpoint_struct_pullback
end
