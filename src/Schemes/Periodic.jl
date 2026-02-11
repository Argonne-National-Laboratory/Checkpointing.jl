# This provides the functionality of periodic checkpointing. It uses the
# terminology of Revolve; see Alg. 799 published by Griewank et al.
# A minor extension is the  optional `bundle` parameter that allows to treat as many loop
# iterations in one tape/adjoint sweep. If `bundle` is 1, the default, then the behavior is that of Alg. 799.

mutable struct Periodic{FT} <: Scheme where {FT}
    steps::Int
    acp::Int
    period::Int
    verbose::Int
    storage::AbstractStorage
    gc::Bool
    chkp_dump::Union{Nothing,ChkpDump}
end

"""
    Periodic(
        checkpoints::Int;
        storage::AbstractStorage = ArrayStorage{MT}(checkpoints),
        verbose::Int = 0,
        gc::Bool = true,
        write_checkpoints::Bool = false,
    ) where {MT}

The periodic scheme is used to store the state of the system at regular intervals
and then restore it when needed.

- `checkpoints`: is the number of checkpoints used for storage.
- `storage`: is the storage backend to use (default is `ArrayStorage`).
- `verbose::Int`: Verbosity level for logging and diagnostics.
- `gc::Bool`: Whether to enable garbage collection (default is `true`).
- `write_checkpoints::Bool`: Whether to enable writing checkpoints (default is `false`).

The period will be `div(steps, checkpoints)`.

"""
function Periodic{FT}(
    steps::Int,
    checkpoints::Int;
    storage::AbstractStorage = ArrayStorage{FT}(checkpoints),
    verbose::Int = 0,
    gc::Bool = true,
    write_checkpoints::Bool = false,
    write_checkpoints_period::Int = 1,
    write_checkpoints_filename::String = "chkp",
) where {FT}
    acp = checkpoints
    period = div(steps, checkpoints)
    if verbose > 0
        @info "[Checkpointing] Periodic checkpointing with $acp checkpoints and period $period"
    end

    Periodic{FT}(
        steps,
        acp,
        period,
        verbose,
        storage,
        gc,
        ChkpDump(
            steps,
            Val(write_checkpoints),
            write_checkpoints_period,
            write_checkpoints_filename,
        ),
    )
end

function Periodic(checkpoints::Integer; storage::Symbol = :ArrayStorage, kwargs...)
    return Periodic{Nothing}(
        0,
        checkpoints;
        storage = eval(storage){Nothing}(checkpoints),
        kwargs...,
    )
end

function instantiate(::Type{FT}, periodic::Periodic{Nothing}, steps::Int) where {FT}
    write_checkpoints = false
    write_checkpoints_period = 1
    write_checkpoints_filename = "chkp"

    if !isa(periodic.chkp_dump, Nothing)
        write_checkpoints = true
        write_checkpoints_period = periodic.chkp_dump.period
        write_checkpoints_filename = periodic.chkp_dump.filename
    end

    checkpoints = min(periodic.acp, steps)
    if checkpoints < periodic.acp
        @warn "Number of checkpoints ($(periodic.acp)) exceeds number of steps ($steps). Using $checkpoints checkpoints."
    end
    return Periodic{FT}(
        steps,
        checkpoints;
        verbose = periodic.verbose,
        storage = similar(periodic.storage, FT),
        gc = periodic.gc,
        write_checkpoints = write_checkpoints,
        write_checkpoints_period = write_checkpoints_period,
        write_checkpoints_filename = write_checkpoints_filename,
    )
end

forwardcount(::Periodic{Nothing}) = nothing

function forwardcount(periodic::Periodic)
    if periodic.acp < 0
        error("Periodic forwardcount: error: checkpoints < 0")
    elseif periodic.steps < 1
        error("Periodic forwardcount: error: steps < 1")
    end
end

function rev_checkpoint_for(
    config,
    body_input::Function,
    dbody::Function,
    alg::Periodic{FT},
    range,
) where {FT}
    body = deepcopy(body_input)
    model_check_outer = alg.storage
    model_check_inner = ArrayStorage{FT}(alg.period)
    if !alg.gc
        GC.enable(false)
    end
    for i = 1:alg.acp
        save!(model_check_outer, deepcopy(body), i)
        for j = ((i-1)*alg.period):((i)*alg.period-1)
            body(j)
        end
    end

    for i = alg.acp:-1:1
        body = deepcopy(load(body, model_check_outer, i))
        for j = 1:alg.period
            save!(model_check_inner, deepcopy(body), j)
            body(j)
        end
        for j = alg.period:-1:1
            dump_prim(alg.chkp_dump, j, body)
            body = deepcopy(load(body, model_check_inner, j))
            Enzyme.autodiff(
                EnzymeCore.set_runtime_activity(Reverse, config),
                Duplicated(body, dbody),
                Const,
                Const(j),
            )
            dump_adj(alg.chkp_dump, j, dbody)
            if !alg.gc
                GC.gc()
            end
        end
    end
    if !alg.gc
        GC.enable(true)
    end
    return nothing
end
