# This provides the functionality of periodic checkpointing. It uses the
# terminology of Revolve; see Alg. 799 published by Griewank et al.
# A minor extension is the  optional `bundle` parameter that allows to treat as many loop
# iterations in one tape/adjoint sweep. If `bundle` is 1, the default, then the behavior is that of Alg. 799.

mutable struct Periodic{FT} <: Scheme
    steps::Int
    acp::Int
    period::Int
    verbose::Int
    storage::AbstractStorage
    chkp_dump::Union{Nothing,ChkpDump}
end

"""
    Periodic(
        checkpoints::Int;
        storage::AbstractStorage = ArrayStorage{MT}(checkpoints),
        verbose::Int = 0,
        write_checkpoints::Bool = false,
    ) where {MT}

The periodic scheme is used to store the state of the system at regular intervals
and then restore it when needed.

- `checkpoints`: is the number of checkpoints used for storage.
- `storage`: is the storage backend to use (default is `ArrayStorage`).
- `verbose::Int`: Verbosity level for logging and diagnostics.
- `write_checkpoints::Bool`: Whether to enable writing checkpoints (default is `false`).

The loop is split into `checkpoints` segments of `div(steps, checkpoints)` or
`cld(steps, checkpoints)` steps each, so every step is covered even when
`checkpoints` does not divide `steps`. `period` is the length of the longest
segment.

"""
function Periodic{FT}(
    steps::Int,
    checkpoints::Int;
    storage::AbstractStorage = ArrayStorage{FT}(checkpoints),
    verbose::Int = 0,
    write_checkpoints::Bool = false,
    write_checkpoints_period::Int = 1,
    write_checkpoints_filename::String = "chkp",
) where {FT}
    acp = checkpoints
    # `div` here used to drop the last `steps % checkpoints` steps from the
    # reverse sweep entirely. Segments now tile the loop (see `_segment`).
    period = checkpoints == 0 ? 0 : cld(steps, checkpoints)
    if verbose > 0
        @info "[Checkpointing] Periodic checkpointing with $acp checkpoints and period $period"
    end

    Periodic{FT}(
        steps,
        acp,
        period,
        verbose,
        storage,
        ChkpDump(
            steps,
            Val(write_checkpoints),
            write_checkpoints_period,
            write_checkpoints_filename,
        ),
    )
end

function Periodic(checkpoints::Integer; storage = ArrayStorage, kwargs...)
    return Periodic{Nothing}(
        0,
        checkpoints;
        storage = _new_storage(storage, checkpoints),
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

# Segment `k` of `acp` covers these positions in the loop. Segment lengths differ
# by at most one, and together the segments tile `1:steps` exactly.
_segment(alg::Periodic, k) = (div((k-1)*alg.steps, alg.acp)+1):div(k*alg.steps, alg.acp)

"""
    fwd_checkpoint_for(body, alg::Periodic, range) -> tape

The primal half of periodic checkpointing: runs the loop on `body` itself,
storing the state at the start of every segment. This is the sweep the reverse
pass used to redo from the initial state.
"""
function fwd_checkpoint_for(body::Function, alg::Periodic, range)
    @assert alg.steps == length(range)
    alg.acp == 0 && return nothing
    for k = 1:alg.acp
        save!(alg.storage, body, k)
        for j in _segment(alg, k)
            body(range[j])
        end
    end
    # A working copy for the reverse sweep, which restores into it before use.
    return (checkpoint_alloc(body),)
end

function rev_checkpoint_for(
    config,
    tape,
    dbody::Function,
    alg::Periodic{FT},
    range,
) where {FT}
    tape === nothing && return nothing
    (body,) = tape
    model_check_outer = alg.storage
    model_check_inner = ArrayStorage{FT}(alg.period)
    for k = alg.acp:-1:1
        load!(body, model_check_outer, k)
        seg = _segment(alg, k)
        for n in eachindex(seg)
            save!(model_check_inner, body, n)
            # The last step of the segment is adjoined straight from its
            # checkpoint below, so running it forward here would be wasted.
            n < length(seg) && body(range[seg[n]])
        end
        # `reverse` alone is EnzymeRules.reverse inside this module.
        for n in Base.reverse(eachindex(seg))
            j = seg[n]
            load!(body, model_check_inner, n)
            dump_prim(alg.chkp_dump, j, body)
            Enzyme.autodiff(
                EnzymeCore.set_runtime_activity(Reverse, config),
                Duplicated(body, dbody),
                Const,
                Const(range[j]),
            )
            dump_adj(alg.chkp_dump, j, dbody)
        end
    end
    return nothing
end
