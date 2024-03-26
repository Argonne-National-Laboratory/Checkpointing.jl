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
    gc::Bool
    write_checkpoints::Bool
end

function Periodic{MT}(
    steps::Int,
    checkpoints::Int,
    fstore::Union{Function,Nothing} = nothing,
    frestore::Union{Function,Nothing} = nothing;
    storage::AbstractStorage = ArrayStorage{MT}(checkpoints),
    anActionInstance::Union{Nothing,Action} = nothing,
    bundle_::Union{Nothing,Int} = nothing,
    verbose::Int = 0,
    gc::Bool = true,
    write_checkpoints::Bool = false
) where {MT}
    if !isa(anActionInstance, Nothing)
        # same as default init above
        anActionInstance.actionflag = 0
        anActionInstance.iteration  = 0
        anActionInstance.cpNum      = 0
    end
    acp             = checkpoints
    period          = div(steps, checkpoints)

    periodic = Periodic{MT}(
        steps, acp, period,verbose,
        fstore, frestore, storage, gc,
        write_checkpoints
    )

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

function rev_checkpoint_struct_for(
    body::Function,
    alg::Periodic,
    model_input::MT,
    shadowmodel::MT,
    range
) where{MT}
    model = deepcopy(model_input)
    model_final = []
    model_check_outer = alg.storage
    model_check_inner = Array{MT}(undef, alg.period)
    check = 0
    if !alg.gc
        GC.enable(false)
    end
    if alg.write_checkpoints
        prim_output = HDF5Storage{MT}(alg.steps; filename="primal_$(alg.write_checkpoints_filename).h5")
        adj_output = HDF5Storage{MT}(alg.steps; filename="adjoint_$(alg.write_checkpoints_filename).h5")
    end
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
            if alg.write_checkpoints && step % alg.write_checkpoints_period == 1
                prim_output[j] = model
            end
            model = deepcopy(model_check_inner[j])
            Enzyme.autodiff(Reverse, body, Duplicated(model,shadowmodel))
            if alg.write_checkpoints && step % alg.write_checkpoints_period == 1
                adj_output[j] = shadowmodel
            end
            if !alg.gc
                GC.gc()
            end
        end
    end
    if alg.write_checkpoints
        close(prim_output.fid)
        close(adj_output.fid)
    end
    if !alg.gc
        GC.enable(true)
    end
    return model_final
end
