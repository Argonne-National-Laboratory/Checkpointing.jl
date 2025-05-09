mutable struct Revolve{FT} <: Scheme where {FT}
    steps::Int
    tail::Int
    acp::Int
    cstart::Int
    cend::Int
    numfwd::Int
    numinv::Int
    numstore::Int
    rwcp::Int
    prevcend::Int
    firstuturned::Bool
    stepof::Vector{Int}
    verbose::Int
    storage::AbstractStorage
    gc::Bool
    chkp_dump::Union{Nothing,ChkpDump}
end

"""
    Revolve{MT}(
        steps::Int, checkpoints::Int;
        storage::AbstractStorage = ArrayStorage{MT}(checkpoints),
        verbose::Int = 0,
        gc::Bool = true,
        write_checkpoints::Bool = false,
        write_checkpoints_period::Int = 1,
        write_checkpoints_filename::String = "chkp",
    ) where {MT}

Creates a new `Revolve` object for checkpointing.
- `steps`: is the number of iterations to perform.
- `checkpoints`: is the number of checkpoints used for storage.
- `storage`: is the storage backend to use (default is `ArrayStorage`).
- `verbose::Int`: Verbosity level for logging and diagnostics.
- `gc::Bool`: Whether to enable garbage collection (default is `true`).
- `write_checkpoints::Bool`: Whether to enable writing checkpoints (default is `false`).
- `write_checkpoints_period::Int`: The period for writing checkpoints (default is `1`).
- `write_checkpoints_filename::String`: The filename for writing checkpoints (default is `"chkp"`).

# References

- Griewank, A. & Walther, A. “Algorithm 799: Revolve: An Implementation of
Checkpointing for the Reverse or Adjoint Mode of Computational Differentiation.”
ACM Transactions on Mathematical Software.

This documentation outlines the structure, usage, and functionality of `Revolve`
and should help users integrate the checkpointing scheme into their Julia
projects.

"""
function Revolve{FT}(
    steps::Int,
    checkpoints::Int;
    storage::AbstractStorage = ArrayStorage{FT}(checkpoints),
    verbose::Int = 0,
    gc::Bool = true,
    write_checkpoints::Bool = false,
    write_checkpoints_period::Int = 1,
    write_checkpoints_filename::String = "chkp",
) where {FT}
    if verbose > 0
        @info "[Checkpointing] Number of checkpoints: $checkpoints"
        @info "[Checkpointing] Number of steps: $steps"
    end
    cstart = 0
    tail = 1
    cend = steps
    acp = checkpoints
    numfwd = 0
    numinv = 0
    numstore = 0
    rwcp = -1
    prevcend = 0
    firstuturned = false
    stepof = Vector{Int}(undef, acp + 1)

    revolve = Revolve{FT}(
        steps,
        tail,
        acp,
        cstart,
        cend,
        numfwd,
        numinv,
        numstore,
        rwcp,
        prevcend,
        firstuturned,
        stepof,
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

    if verbose > 0
        predfwdcnt = forwardcount(revolve)
        if predfwdcnt == -1
            error("Revolve: error returned by  revolve::forwardcount")
        else
            @info "[Checkpointing] Prediction:"
            @info "[Checkpointing] Forward steps   : $(Int(predfwdcnt))"
            @info "[Checkpointing] Overhead factor : $(predfwdcnt/(steps))"
        end
    end
    return revolve
end

function next_action!(revolve::Revolve)::Action
    # Default values for next action
    actionflag = none
    iteration = 0
    startiteration = 0
    cpnum = 0
    if revolve.numinv == 0
        # first invocation
        for v in revolve.stepof
            v = 0
        end
        revolve.stepof[1] = revolve.cstart - 1
    end
    prevcstart = revolve.cstart
    revolve.numinv += 1
    rwcptest = (revolve.rwcp == -1)
    if !rwcptest
        rwcptest = revolve.stepof[revolve.rwcp+1] != revolve.cstart
    end
    if (revolve.cend - revolve.cstart) == 0
        # nothing in current subrange
        if (revolve.rwcp == -1) || (revolve.cstart == revolve.stepof[1])
            # we are done
            revolve.rwcp = revolve.rwcp - 1
            if revolve.verbose > 0
                @info "[Checkpointing] Summary:"
                @info "[Checkpointing] Forward steps:    $(revolve.numfwd)"
                @info "[Checkpointing] CP stores:        $(revolve.numstore)"
                @info "[Checkpointing] NextAction calls: $(revolve.numinv)"
            end
            actionflag = done
        else
            revolve.cstart = revolve.stepof[revolve.rwcp+1]
            revolve.prevcend = revolve.cend
            actionflag = restore
        end
    elseif (revolve.cend - revolve.cstart) == 1
        revolve.cend = revolve.cend - 1
        revolve.prevcend = revolve.cend
        if (revolve.rwcp >= 0) && (revolve.stepof[revolve.rwcp+1] == revolve.cstart)
            revolve.rwcp -= 1
        end
        if !revolve.firstuturned
            actionflag = firstuturn
            revolve.firstuturned = true
        else
            actionflag = uturn
        end
    elseif rwcptest
        revolve.rwcp += 1
        if revolve.rwcp + 1 > revolve.acp
            error("Revolve: insufficient allowed checkpoints")
        else
            revolve.stepof[revolve.rwcp+1] = revolve.cstart
            revolve.numstore += 1
            revolve.prevcend = revolve.cend
            actionflag = store
        end
    elseif (revolve.prevcend < revolve.cend) && (revolve.acp == revolve.rwcp + 1)
        error("Revolve: insufficient allowed checkpoints")
    else
        availcp = revolve.acp - revolve.rwcp
        if availcp < 1
            error("Revolve: insufficient allowed checkpoints")
        else
            reps = 0
            range = 1
            while range < (revolve.cend - revolve.cstart)
                reps = reps + 1
                range = range * (reps + availcp) / reps
            end
            bino1 = range * reps / (availcp + reps)
            if availcp > 1
                bino2 = bino1 * availcp / (availcp + reps - 1)
            else
                bino2 = 1
            end
            if availcp == 1
                bino3 = 0
            elseif availcp > 2
                bino3 = bino2 * (availcp - 1) / (availcp + reps - 2)
            else
                bino3 = 1
            end
            bino4 = bino2 * (reps - 1) / availcp
            if availcp < 3
                bino5 = 0
            elseif availcp > 3
                bino5 = bino3 * (availcp - 1) / reps
            else
                bino5 = 1
            end
            if (revolve.cend - revolve.cstart) <= (bino1 + bino3)
                revolve.cstart = trunc(Int, revolve.cstart + bino4)
            elseif (revolve.cend - revolve.cstart) >= (range - bino5)
                revolve.cstart = trunc(Int, revolve.cstart + bino1)
            else
                revolve.cstart = trunc(Int, revolve.cend - bino2 - bino3)
            end
            if revolve.cstart == prevcstart
                revolve.cstart = prevcstart + 1
            end
            if revolve.cstart == revolve.steps
                revolve.numfwd =
                    (revolve.numfwd + ((revolve.cstart - 1) - prevcstart) + revolve.tail)
            else
                revolve.numfwd = revolve.numfwd + revolve.cstart - prevcstart
            end
            actionflag = forward
        end
    end
    startiteration = prevcstart
    if actionflag == firstuturn
        iteration = revolve.cstart + revolve.tail
    elseif actionflag == uturn
        iteration = revolve.cstart + 1
    else
        iteration = revolve.cstart
    end
    if revolve.verbose > 2
        if actionflag == forward
            @info "[Checkpointing] Run forward iterations    [$startiteration, $(iteration - 1)]"
        elseif actionflag == restore
            @info "[Checkpointing] Restore input of iteration $iteration"
        elseif actionflag == firstuturn

            @info "[Checkpointing] 1st uturn for iterations  [$startiteration, $(iteration - 1)]"
        elseif actionflag == uturn
            @info "[Checkpointing] Uturn for iterations      [$startiteration, $(iteration - 1)]"
        end
    end
    if (revolve.verbose > 1) && (actionflag == store)
        @info "[Checkpointing] Store input of iteration $iteration  "
    end
    cpnum = revolve.rwcp

    return Action(actionflag, iteration, startiteration, cpnum)

end

function guess(revolve::Revolve)::Int
    bSteps = revolve.steps
    if revolve.steps < 1
        error("Revolve: error: steps < 1")
    else
        if bSteps == 1
            guess = 0
        else
            checkpoints = 1
            reps = 1
            s = 0
            while chkrange(revolve, checkpoints + s, reps + s) > bSteps
                s -= 1
            end
            while chkrange(revolve, checkpoints + s, reps + s) < bSteps
                s += 1
            end
            checkpoints += s
            reps += s
            s = -1
            while chkrange(revolve, checkpoints, reps) >= bSteps
                if checkpoints > reps
                    checkpoints -= 1
                    s = 0
                else
                    reps -= 1
                    s = 1
                end
            end
            if s == 0
                checkpoints += 1
            end
            if s == 1
                reps += 1
            end
            guess = checkpoints
        end
    end
    return guess
end

function factor(revolve::Revolve, steps, checkpoints)
    f = forwardcount(revolve)
    if f == -1
        error("Revolve: error returned by forwardcount")
    else
        factor = float(f) / steps
    end
    return factor
end

function chkrange(::Revolve, ss, tt)
    ret = Int(0)
    res = 1.0
    if tt < 0 || ss < 0
        error("Revolve chkrange: error: negative parameter")
    else
        for i = 1:tt
            res = res * (ss + i)
            res = res / i
            if res > typemax(typeof(ret))
                break
            end
        end
        if res < typemax(typeof(ret))
            ret = trunc(Int, res)
        else
            ret = typemax(typeof(ret))
            @warn "Revolve chkrange: warning: returning maximal integer "
        end
    end
    return ret
end

function forwardcount(revolve::Revolve)
    checkpoints = revolve.acp
    steps = revolve.steps
    if checkpoints < 0
        error("Revolve forwardcount: error: checkpoints < 0")
    elseif steps < 1
        error("Revolve forwardcount: error: steps < 1")
    else
        s = steps
        if s == 1
            ret = 0
        elseif checkpoints == 0
            error("Revolve forarwdCount: error: given inputs require checkpoints > 0")
        else
            reps = 0
            range = 1
            while range < s
                reps = reps + 1
                range = range * (reps + checkpoints) / reps
            end
            ret = reps * s - range * reps / (checkpoints + 1)
        end
    end
    return ret
end

function reset!(revolve::Revolve)
    revolve.cstart = 0
    revolve.tail = 1
    revolve.numfwd = 0
    revolve.numinv = 0
    revolve.numstore = 0
    revolve.rwcp = -1
    revolve.prevcend = 0
    revolve.firstuturned = false
    return nothing
end

function rev_checkpoint_struct_for(
    config,
    body_input::Function,
    dbody::Function,
    alg::Revolve{FT},
    range,
) where {FT}
    body = deepcopy(body_input)
    if alg.verbose > 0
        @info "[Checkpointing] Size per checkpoint: $(Base.format_bytes(Base.summarysize(dbody)))"
    end
    storemap = Dict{Int32,Int32}()
    check = 0
    model_check = alg.storage
    if !alg.gc
        GC.enable(false)
    end
    step = alg.steps
    while true
        next_action = next_action!(alg)
        if (next_action.actionflag == Checkpointing.store)
            check = check + 1
            storemap[next_action.iteration-1] = check
            model_check[check] = deepcopy(body)
        elseif (next_action.actionflag == Checkpointing.forward)
            for j = next_action.startiteration:(next_action.iteration-1)
                body()
            end
        elseif (next_action.actionflag == Checkpointing.firstuturn)
            # body()
            # dump_prim(alg.chkp_dump, step, model_final)
            if alg.verbose > 0
                @info "[Checkpointing] First uturn"
                @info "[Checkpointing] Size of total storage: $(Base.format_bytes(Base.summarysize(alg.storage)))"
            end
            # println("A in:", body.A)
            # println("y in:", body.y)
            # println("dA in:", dbody.A)
            # println("dy in:", dbody.y)
            # @show typeof(body)
            # @show fieldname
            Enzyme.autodiff(
                EnzymeCore.set_runtime_activity(Reverse, config),
                # ReverseWithPrimal,
                Duplicated(body, dbody),
                Const,
                # Duplicated(model, dmodel),
            )
            # println("dA out:", dbody.A)
            # println("dy out:", dbody.y)
            # dump_adj(alg.chkp_dump, step, dmodel)
            step -= 1
            if !alg.gc
                GC.gc()
            end
        elseif (next_action.actionflag == Checkpointing.uturn)
            # dump_prim(alg.chkp_dump, step, model)
            # println("dA in:", dbody.A)
            Enzyme.autodiff(
                EnzymeCore.set_runtime_activity(Reverse, config),
                Duplicated(body, dbody),
                Const,
            )
            # println("dA out:", dbody.A)
            # dump_adj(alg.chkp_dump, step, dmodel)
            step -= 1
            if !alg.gc
                GC.gc()
            end
            if haskey(storemap, next_action.iteration - 1 - 1)
                delete!(storemap, next_action.iteration - 1 - 1)
                check = check - 1
            end
        elseif (next_action.actionflag == Checkpointing.restore)
            body = deepcopy(model_check[storemap[next_action.iteration-1]])
        elseif next_action.actionflag == Checkpointing.done
            if haskey(storemap, next_action.iteration - 1 - 1)
                delete!(storemap, next_action.iteration - 1 - 1)
                check = check - 1
            end
            break
        end
    end
    if !alg.gc
        GC.enable(true)
    end
    return nothing
end
