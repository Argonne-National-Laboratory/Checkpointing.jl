"""
    Revolve

This is a Julia adaptation of the functionality of Revolve; see Alg. 799 published by Griewank et al.
A minor extension is the  optional `bundle` parameter that allows to treat as many loop
iterations in one tape/adjoint sweep. If `bundle` is 1, the default, then the behavior is that of Alg. 799.

"""
mutable struct Revolve{MT} <: Scheme where {MT}
    steps::Int
    bundle::Int
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
    fstore::Union{Function,Nothing}
    frestore::Union{Function,Nothing}
    storage::AbstractStorage
    gc::Bool
    chkp_dump::Union{Nothing,ChkpDump}
end

function Revolve{MT}(
    steps::Int,
    checkpoints::Int,
    fstore::Union{Function,Nothing} = nothing,
    frestore::Union{Function,Nothing} = nothing;
    storage::AbstractStorage = ArrayStorage{MT}(checkpoints),
    anActionInstance::Union{Nothing,Action} = nothing,
    bundle_::Union{Nothing,Int} = nothing,
    verbose::Int = 0,
    gc::Bool = true,
    write_checkpoints::Bool = false,
    write_checkpoints_period::Int = 1,
    write_checkpoints_filename::String = "chkp",
) where {MT}
    if !isa(anActionInstance, Nothing)
        # same as default init above
        anActionInstance.actionflag = 0
        anActionInstance.iteration = 0
        anActionInstance.cpNum = 0
    end
    if verbose > 0
        @info "Revolve: Number of checkpoints: $checkpoints"
        @info "Revolve: Number of steps: $steps"
    end
    !isa(bundle_, Nothing) ? bundle = bundle_ : bundle = 1
    if bundle < 1 || bundle > steps
        error("Revolve: bundle parameter out of range [1,steps]")
    elseif steps < 0
        error("Revolve: negative steps")
    elseif checkpoints < 0
        error("Revolve: negative checkpoints")
    end
    cstart = 0
    tail = 1
    if bundle > 1
        tail = mod(steps, bundle)
        steps = steps / bundle
        if tail > 0
            step += 1
        else
            tail = bundle
        end
    end
    cend = steps
    acp = checkpoints
    numfwd = 0
    numinv = 0
    numstore = 0
    rwcp = -1
    prevcend = 0
    firstuturned = false
    stepof = Vector{Int}(undef, acp + 1)

    revolve = Revolve{MT}(
        steps,
        bundle,
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
        fstore,
        frestore,
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
            @info "Prediction:"
            @info "Forward steps   : $(Int(predfwdcnt))"
            @info "Overhead factor : $(predfwdcnt/(steps))"
        end
    end
    return revolve
end

function adjust(::Revolve)
    error("Not implemented")
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
            if revolve.verbose > 2
                @info "Done"
            end
            if revolve.verbose > 0
                @info "Summary:"
                @info " Forward steps: $(revolve.numfwd)"
                @info " CP stores             : $(revolve.numstore)"
                @info " NextAction calls      : $(revolve.numinv)"
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
                revolve.numfwd = (
                    revolve.numfwd +
                    ((revolve.cstart - 1) - prevcstart) * revolve.bundle +
                    revolve.tail
                )
            else
                revolve.numfwd =
                    revolve.numfwd + (revolve.cstart - prevcstart) * revolve.bundle
            end
            actionflag = forward
        end
    end
    startiteration = prevcstart * revolve.bundle
    if actionflag == firstuturn
        iteration = revolve.cstart * revolve.bundle + revolve.tail
    elseif actionflag == uturn
        iteration = (revolve.cstart + 1) * revolve.bundle
    else
        iteration = revolve.cstart * revolve.bundle
    end
    if revolve.verbose > 2
        if actionflag == forward
            @info " run forward iterations    [$startiteration, $(iteration - 1)]"
        elseif actionflag == restore
            @info " restore input of iteration $iteration"
        elseif actionflag == firstuturn

            @info " 1st uturn for iterations  [$startiteration, $(iteration - 1)]"
        elseif actionflag == uturn
            @info " uturn for iterations      [$startiteration, $(iteration - 1)]"
        end
    end
    if (revolve.verbose > 1) && (actionflag == store)
        @info " store input of iteration $iteration  "
    end
    cpnum = revolve.rwcp

    return Action(actionflag, iteration, startiteration, cpnum)

end

function guess(revolve::Revolve; bundle::Union{Nothing,Int} = nothing)::Int
    b = 1
    bSteps = revolve.steps
    if !isa(bundle, Nothing)
        b = bundle
    end
    if revolve.steps < 1
        error("Revolve: error: steps < 1")
    elseif b < 1
        error("Revolve: error: bundle < 1")
    else
        if b > 1
            revolve.tail = mod(bSteps, b)
            bSteps = div(bSteps, b)
            if revolve.tail > 0
                bSteps = bSteps + 1
            end
        end
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

function factor(revolve::Revolve, steps, checkpoints, bundle::Union{Nothing,Int} = nothing)
    b = 1
    if !isa(bundle, Nothing)
        b = bundle
    end
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
    bundle = revolve.bundle
    steps = revolve.steps
    if checkpoints < 0
        error("Revolve forwardcount: error: checkpoints < 0")
    elseif steps < 1
        error("Revolve forwardcount: error: steps < 1")
    elseif bundle < 1
        error("Revolve forwardcount: error: bundle < 1")
    else
        s = steps
        if bundle > 1
            tail = mod(s, bundle)
            s = s / bundle
            if tail > 0
                s = s + 1
            end
        end
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
            ret = (reps * s - range * reps / (checkpoints + 1)) * bundle
        end
    end
    return ret
end

function reset!(revolve::Revolve)
    revolve.cstart = 0
    revolve.tail = 1
    if revolve.bundle > 1
        tail = mod(steps, bundle)
        steps = steps / bundle
        if tail > 0
            step += 1
        else
            tail = bundle
        end
    end
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
    body::Function,
    alg::Revolve,
    model_input::MT,
    shadowmodel::MT,
    range,
) where {MT}
    model = deepcopy(model_input)
    if alg.verbose > 0
        @info "Size per checkpoint: $(Base.format_bytes(Base.summarysize(model)))"
    end
    storemap = Dict{Int32,Int32}()
    check = 0
    model_check = alg.storage
    model_final = []
    if !alg.gc
        GC.enable(false)
    end
    step = alg.steps
    while true
        next_action = next_action!(alg)
        if (next_action.actionflag == Checkpointing.store)
            check = check + 1
            storemap[next_action.iteration-1] = check
            model_check[check] = deepcopy(model)
        elseif (next_action.actionflag == Checkpointing.forward)
            for j = next_action.startiteration:(next_action.iteration-1)
                body(model)
            end
        elseif (next_action.actionflag == Checkpointing.firstuturn)
            body(model)
            model_final = deepcopy(model)
            dump_prim(alg.chkp_dump, step, model_final)
            if alg.verbose > 0
                @info "Revolve: First Uturn"
                @info "Size of total storage: $(Base.format_bytes(Base.summarysize(alg.storage)))"
            end
            Enzyme.autodiff(EnzymeCore.set_runtime_activity(Reverse, config), Const(body), Duplicated(model, shadowmodel))
            dump_adj(alg.chkp_dump, step, shadowmodel)
            step -= 1
            if !alg.gc
                GC.gc()
            end
        elseif (next_action.actionflag == Checkpointing.uturn)
            dump_prim(alg.chkp_dump, step, model)
            Enzyme.autodiff(EnzymeCore.set_runtime_activity(Reverse, config), Const(body), Duplicated(model, shadowmodel))
            dump_adj(alg.chkp_dump, step, shadowmodel)
            step -= 1
            if !alg.gc
                GC.gc()
            end
            if haskey(storemap, next_action.iteration - 1 - 1)
                delete!(storemap, next_action.iteration - 1 - 1)
                check = check - 1
            end
        elseif (next_action.actionflag == Checkpointing.restore)
            model = deepcopy(model_check[storemap[next_action.iteration-1]])
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
