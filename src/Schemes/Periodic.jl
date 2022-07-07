# This provides the functionality of periodic checkpointing. It uses the
# terminology of Revolve; see Alg. 799 published by Griewank et al.
# A minor extension is the  optional `bundle` parameter that allows to treat as many loop
# iterations in one tape/adjoint sweep. If `bundle` is 1, the default, then the behavior is that of Alg. 799.

"""
    Periodic

Periodic checkpointing scheme.

"""
mutable struct Periodic{MT} <: Scheme where {MT}
    steps::Int
    acp::Int
    period::Int
    verbose::Int
    fstore::Union{Function,Nothing}
    frestore::Union{Function,Nothing}
    storage::AbstractStorage
end

function Periodic{MT}(
    steps::Int,
    checkpoints::Int,
    fstore::Union{Function,Nothing} = nothing,
    frestore::Union{Function,Nothing} = nothing;
    storage::AbstractStorage = ArrayStorage{MT}(checkpoints),
    anActionInstance::Union{Nothing,Action} = nothing,
    bundle_::Union{Nothing,Int} = nothing,
    verbose::Int = 0
) where {MT}
    if !isa(anActionInstance, Nothing)
        # same as default init above
        anActionInstance.actionflag = 0
        anActionInstance.iteration  = 0
        anActionInstance.cpNum      = 0
    end
    acp             = checkpoints
    period          = div(steps, checkpoints)

    periodic = Periodic{MT}(steps, acp, period, verbose, fstore, frestore, storage)

    forwardcount(periodic)
    return periodic
end

function forwardcount(periodic::Periodic)
    if periodic.acp < 0
        error("Periodic forwardcount: error: checkpoints < 0")
    elseif periodic.steps < 1
        error("Periodic forwardcount: error: steps < 1")
    elseif mod(periodic.steps, periodic.acp) != 0
        error("Periodic forwardcount: error: steps ", periodic.steps, " not divisible by checkpoints ", periodic.acp)
    end
end

function checkpoint_struct(body::Function,
        alg::Periodic,
        model_input::MT,
        shadowmodel::MT
    ) where{MT}
    model = deepcopy(model_input)
    model_final = []
    model_check_outer = alg.storage
    model_check_inner = Array{MT}(undef, alg.period)
    check = 0
    for i = 1:alg.acp
        model_check_outer[i] = deepcopy(model)
        for j= (i-1)*alg.period: (i)*alg.period-1
            body(model)
        end
    end
    model_final = deepcopy(model)
    for i = alg.acp:-1:1
        model = deepcopy(model_check_outer[i])
        for j= 1:alg.period
            model_check_inner[j] = deepcopy(model)
            body(model)
        end
        for j= alg.period:-1:1
            model = deepcopy(model_check_inner[j])
            Enzyme.autodiff(body, Duplicated(model,shadowmodel))
        end
    end
    return model_final
end
