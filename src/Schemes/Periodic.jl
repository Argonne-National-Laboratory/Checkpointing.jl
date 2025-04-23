# This provides the functionality of periodic checkpointing. It uses the
# terminology of Revolve; see Alg. 799 published by Griewank et al.
# A minor extension is the  optional `bundle` parameter that allows to treat as many loop
# iterations in one tape/adjoint sweep. If `bundle` is 1, the default, then the behavior is that of Alg. 799.

mutable struct Periodic{MT} <: Scheme where {MT}
    steps::Int
    acp::Int
    period::Int
    verbose::Int
    storage::AbstractStorage
    gc::Bool
    write_checkpoints::Bool
end

"""
    Periodic(
        steps::Int,
        checkpoints::Int;
        storage::AbstractStorage = ArrayStorage{MT}(checkpoints),
        verbose::Int = 0,
        gc::Bool = true,
        write_checkpoints::Bool = false,
    ) where {MT}

The periodic scheme is used to store the state of the system at regular intervals
and then restore it when needed.

- `steps`: is the number of iterations to perform.
- `checkpoints`: is the number of checkpoints used for storage.
- `storage`: is the storage backend to use (default is `ArrayStorage`).
- `verbose::Int`: Verbosity level for logging and diagnostics.
- `gc::Bool`: Whether to enable garbage collection (default is `true`).
- `write_checkpoints::Bool`: Whether to enable writing checkpoints (default is `false`).

The period will be `div(steps, checkpoints)`.

"""
function Periodic{MT}(
    steps::Int,
    checkpoints::Int;
    storage::AbstractStorage = ArrayStorage{MT}(checkpoints),
    verbose::Int = 0,
    gc::Bool = true,
    write_checkpoints::Bool = false,
) where {MT}
    acp = checkpoints
    period = div(steps, checkpoints)
    if verbose > 0
        @info "[Checkpointing] Periodic checkpointing with $acp checkpoints and period $period"
    end

    periodic = Periodic{MT}(steps, acp, period, verbose, storage, gc, write_checkpoints)

    forwardcount(periodic)
    return periodic
end

function forwardcount(periodic::Periodic)
    if periodic.acp < 0
        error("Periodic forwardcount: error: checkpoints < 0")
    elseif periodic.steps < 1
        error("Periodic forwardcount: error: steps < 1")
    end
end

function rev_checkpoint_struct_for(
    config,
    body::Function,
    alg::Periodic,
    model_input::MT,
    shadowmodel::MT,
    range,
) where {MT}
    model = deepcopy(model_input)
    model_final = []
    model_check_outer = alg.storage
    model_check_inner = Array{MT}(undef, alg.period)
    if !alg.gc
        GC.enable(false)
    end
    if alg.write_checkpoints
        prim_output = HDF5Storage{MT}(
            alg.steps;
            filename = "primal_$(alg.write_checkpoints_filename).h5",
        )
        adj_output = HDF5Storage{MT}(
            alg.steps;
            filename = "adjoint_$(alg.write_checkpoints_filename).h5",
        )
    end
    for i = 1:alg.acp
        model_check_outer[i] = deepcopy(model)
        for j = ((i-1)*alg.period):((i)*alg.period-1)
            body(model)
        end
    end
    model_final = deepcopy(model)
    for i = alg.acp:-1:1
        model = deepcopy(model_check_outer[i])
        for j = 1:alg.period
            model_check_inner[j] = deepcopy(model)
            body(model)
        end
        for j = alg.period:-1:1
            if alg.write_checkpoints && step % alg.write_checkpoints_period == 1
                prim_output[j] = model
            end
            model = deepcopy(model_check_inner[j])
            Enzyme.autodiff(
                EnzymeCore.set_runtime_activity(Reverse, config),
                Const(body),
                Duplicated(model, shadowmodel),
            )
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
