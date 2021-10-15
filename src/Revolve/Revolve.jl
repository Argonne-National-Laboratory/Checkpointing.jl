# This is a Julia adaptation of the functionality of Revolve; see Alg. 799 published by Griewank et al.
# A minor extension is the  optional `bundle` parameter that allows to treat as many loop
# iterations in one tape/adjoint sweep. If `bundle` is 1, the default, then the behavior is that of Alg. 799.

"""
    Action

    none: no action
    store: store a checkpoint now equivalent to TAKESHOT in Alg. 79
    restore: restore a checkpoint now equivalent to RESTORE in Alg. 79
    forward: execute iteration(s) forward equivalent to ADVANCE in Alg. 79
    firstuturn: tape iteration(s); optionally leave to return later;  and (upon return) do the adjoint(s) equivalent to FIRSTTURN in Alg. 799
    uturn: tape iteration(s) and do the adjoint(s) equivalent to YOUTURN in Alg. 79
    done: we are done with adjoining the loop equivalent to the `terminate` enum value in Alg. 79
"""
@enum ActionFlag begin
    none
	store
	restore
	forward
	firstuturn
	uturn
	done
end

@with_kw struct Action
	actionflag::ActionFlag
	iteration::Int
	startiteration::Int
	cpnum::Int
end

@with_kw mutable struct Revolve <: Scheme
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
end

function Revolve(steps::Int, checkpoints::Int; anActionInstance::Union{Nothing,Action} = nothing, bundle_::Union{Nothing,Int} = nothing, verbose::Int = 0)
    if !isa(anActionInstance, Nothing)
        # same as default init above
        anActionInstance.actionflag = 0
        anActionInstance.iteration  = 0
        anActionInstance.cpNum      = 0
    end
    !isa(bundle_, Nothing) ? bundle = bundle_ : bundle = 1
    if bundle < 1 || bundle > steps
       error("Revolve: bundle parameter out of range [1,steps]")
    elseif steps<0
       error("Revolve: negative steps")
    elseif checkpoints < 0
       error("Revolve: negative checkpoints")
    end
    cstart = 0
    tail   = 1
    if bundle > 1
        tail = mod(steps, bundle)
        steps = steps / bundle
        if tail > 0
            step += 1
        else
            tail = bundle
        end
    end
    cend            = steps
    acp             = checkpoints
    numfwd          = 0
    numinv          = 0
    numstore        = 0
    rwcp            = -1
    prevcend        = 0
    firstuturned    = false
    stepof = Vector{Int}(undef, acp+1)

    if verbose > 0
        predfwdcnt = forwardcount(steps, acp, bundle)
        if predfwdcnt == -1
            error("Revolve: error returned by  revolve::forwardcount")
        else
            @info "prediction:"
            @info " overhead forward steps : $predfwdcnt"
            @info " overhead factor        : $(predfwdcnt/steps)"
        end
    end
    return Revolve(steps, bundle, tail, acp, cstart, cend, numfwd, numinv, numstore, rwcp, prevcend, firstuturned, stepof, verbose)
end

function adjust(::Revolve)
    error("Not implemented")
end

function next_action!(revolve::Revolve)::Action
    # Default values for next action
    actionflag     = none
    iteration      = 0
    startiteration = 0
    cpnum          = 0
    @unpack_Revolve revolve
    if numinv == 0
        # first invocation
        for v in stepof
            v = 0
        end
        stepof[1] = cstart - 1
    end
    prevcstart = cstart
    numinv += 1
    rwcptest = (rwcp == -1)
    if !rwcptest
       rwcptest = stepof[rwcp+1] != cstart
    end
    if (cend - cstart) == 0
       # nothing in current subrange
        if (rwcp == -1) || (cstart == stepof[1])
            # we are done
            rwcp = rwcp - 1
            if verbose > 2
                @info "done"
            end
            if verbose > 0
                @info "summary:"
                @info " overhead forward steps: $numfwd"
                @info " CP stores             : $numstore"
                @info " NextAction calls      : $numinv"
            end
            actionflag = done
        else
           cstart = stepof[rwcp+1]
           prevcend = cend
           actionflag = restore
        end
    elseif (cend - cstart) == 1
        cend = cend - 1
        prevcend = cend
        if (rwcp >= 0) && (stepof[rwcp + 1] == cstart)
            rwcp -= 1
        end
        if !firstuturned
            actionflag = firstuturn
            firstuturned = true
        else
            actionflag = uturn
        end
    elseif rwcptest
        rwcp += 1
        if rwcp + 1 > acp
            error("Revolve: insufficient allowed checkpoints")
        else
            stepof[rwcp+1] = cstart
            numstore += 1
            prevcend = cend
            actionflag = store
        end
    elseif (prevcend < cend) && (acp == rwcp + 1)
            error("Revolve: insufficient allowed checkpoints")
    else
        availcp = acp - rwcp
        if availcp < 1
            error("Revolve: insufficient allowed checkpoints")
        else
            reps = 0
            range = 1
            while range < (cend - cstart)
                reps = reps + 1
                range = range * (reps + availcp) / reps
            end
            bino1 = range * reps / (availcp + reps)
            if availcp > 1
                bino2 = bino1 * availcp / (availcp + reps - 1)
            else
                bino2 = 1
            end
            if availcp==1
                bino3 = 0
            elseif availcp>2
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
            if (cend - cstart) <= (bino1 + bino3)
                cstart = trunc(Int, cstart + bino4)
            elseif (cend - cstart) >= (range-bino5)
                cstart = trunc(Int, cstart + bino1)
            else
                cstart = trunc(Int, cend - bino2 - bino3)
            end
            if cstart == prevcstart
                cstart = prevcstart + 1
            end
            if cstart == steps
                numfwd = numfwd + ((cstart - 1) - prevcstart) * bundle + tail
            else
                numfwd = numfwd + (cstart - prevcstart) * bundle
            end
            actionflag = forward
        end
    end
    startiteration = prevcstart * bundle
    if actionflag == firstuturn
        iteration = cstart * bundle + tail
    elseif actionflag == uturn
        iteration = (cstart + 1) * bundle
    else
        iteration = cstart * bundle
    end
    if verbose > 2
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
    if (verbose > 1) && (actionflag == store)
        @info " store input of iteration $iteration  "
    end
    cpnum=rwcp
    @pack_Revolve! revolve

    return Action(actionflag, iteration, startiteration, cpnum)

end

function guess(revolve::Revolve; bundle::Union{Nothing, Int} = nothing)::Int
    @unpack_Revolve revolve
    b=1
    bSteps=steps
    if !isa(bundle, Nothing)
        b=bundle
    end
    if steps < 1
        error("Revolve: error: steps < 1")
    elseif b<1
        error("Revolve: error: bundle < 1")
    else
        if b > 1
            tail = mod(bSteps, b)
            bSteps = div(bSteps, b)
            if tail > 0
                bSteps = bSteps + 1
            end
        end
        if bSteps == 1
            guess = 0
        else
            checkpoints = 1
            reps = 1
            s = 0
            while chkrange(checkpoints+s, reps+s) > bSteps
                s -= 1
            end
            while chkrange(checkpoints+s, reps+s) < bSteps
                s += 1
            end
            checkpoints += s
            reps += s
            s = -1
            while chkrange(checkpoints, reps) >= bSteps
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

function factor(steps, checkpoints, bundle::Union{Nothing,Int} = nothing)
    b = 1
    if !isa(bundle, Nothing)
        b = bundle
    end
    f = forwardcount(steps, checkpoints, b)
    if f == -1
        error("Revolve: error returned by forwardcount")
    else
        factor = float(f)/steps
    end
    return factor
end

function chkrange(ss, tt)
    ret = Int(0)
    res = 1.
    if tt < 0 || ss < 0
        error("Revolve chkrange: error: negative parameter")
    else
        for i in 1:tt
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

function forwardcount(steps, checkpoints, bundle)
    if checkpoints < 0
        error("Revolve forwardcount: error: checkpoints < 0")
    elseif steps < 1
        error("Revolve forwardcount: error: steps < 1")
    elseif bundle < 1
        error("Revolve forwardcount: error: bundle < 1")
    else
        s=steps
        if bundle > 1
            tail = mod(s,bundle)
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
            range = range*(reps+checkpoints)/reps
            end
            ret = (reps * s - range * reps / (checkpoints + 1)) * bundle
        end
    end
    return ret
end