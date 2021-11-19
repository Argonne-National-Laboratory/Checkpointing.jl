# This provides the efunctionality of periodic checkpointing. It uses the
# terminology of Revolve; see Alg. 799 published by Griewank et al.
# A minor extension is the  optional `bundle` parameter that allows to treat as many loop
# iterations in one tape/adjoint sweep. If `bundle` is 1, the default, then the behavior is that of Alg. 799.

mutable struct Periodic <: Scheme
    steps::Int
    bundle::Int
    tail::Int
    acp::Int
    curriter::Int
    cend::Int
    numfwd::Int
    numinv::Int
    numstore::Int
    rwcp::Int
    prevcend::Int
    period::Int
    firstuturned::Bool
    stepof::Vector{Int}
    verbose::Int
    fstore::Function
    frestore::Function
end

function Periodic(steps::Int, checkpoints::Int, fstore::Function, frestore::Function; anActionInstance::Union{Nothing,Action} = nothing, bundle_::Union{Nothing,Int} = nothing, verbose::Int = 0)
    if !isa(anActionInstance, Nothing)
        # same as default init above
        anActionInstance.actionflag = 0
        anActionInstance.iteration  = 0
        anActionInstance.cpNum      = 0
    end
    bundle = 1
    #=
    !isa(bundle_, Nothing) ? bundle = bundle_ : bundle = 1
    if bundle < 1 || bundle > steps
       error("Periodic: bundle parameter out of range [1,steps]")
    elseif steps<0
       error("Periodic: negative steps")
    elseif checkpoints < 0
       error("Periodic: negative checkpoints")
    end
    =#
    curriter = 0
    tail   = 1
    #=
    if bundle > 1
        tail = mod(steps, bundle)
        steps = steps / bundle
        if tail > 0
            step += 1
        else
            tail = bundle
        end
    end
    =#
    cend            = steps
    acp             = checkpoints
    numfwd          = 0
    numinv          = 0
    numstore        = 0
    rwcp            = -1
    prevcend        = 0
    firstuturned    = false
    stepof = Vector{Int}(undef, acp+1)
    period          = div(steps, checkpoints)

    periodic = Periodic(steps, bundle, tail, acp, curriter, cend, numfwd, numinv, numstore, rwcp, prevcend, period, firstuturned, stepof, verbose, fstore, frestore)

    forwardcount(periodic)
    return periodic
end

function next_action!(periodic::Periodic)::Action
    # Default values for next action
    actionflag     = none
    iteration      = 0
    startiteration = 0
    cpnum          = 0
    storedorrestored = false
    #=if numinv == 0
        # first invocation
        for v in stepof
            v = 0
        end
        stepof[1] = curriter - 1
    end=#
    prevcurriter = periodic.curriter
    periodic.numinv += 1
    if (!periodic.firstuturned)
        if (periodic.curriter == (periodic.acp-1)*periodic.period)
            if (storedorrestored)
                actionflag = firstuturn
                periodic.firstuturned = true
            else
                actionflag = store
                storedorrestored = true
            end
        else
            if (storedorrestored)
                actionflag = forward
            else
                actionflag = store
                storedorrestored = true
                periodic.curriter == periodic.curriter + periodic.period
            end
        end
    else
        if (periodic.curriter == 0) && (actionflag == adjoint)
            actionflag = done
        elseif (periodic.curriter == 0)
            if (storedorrestored)
                actionflag = uturn
                storedorrestored = false
            else
                actionflag = restore
                storedorrestored = true
                periodic.curriter == periodic.curriter - periodic.period
            end
        else
            if (storedorrestored)
                actionflag = uturn
                storedorrestored = false
            else
                actionflag = restore
                storedorrestored = true
            end
        end
    end
    startiteration = prevcurriter * periodic.bundle
    if actionflag == firstuturn
        iteration = periodic.curriter * periodic.bundle + periodic.tail
    elseif actionflag == uturn
        iteration = (periodic.curriter + 1) * periodic.bundle
    else
        iteration = periodic.curriter * periodic.bundle
    end
    if periodic.verbose > 2
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
    if (periodic.verbose > 1) && (actionflag == store)
        @info " store input of iteration $iteration  "
    end
    cpnum=periodic.rwcp

    return Action(actionflag, iteration, startiteration, cpnum)

end

function forwardcount(periodic::Periodic)
    if periodic.acp < 0
        error("Periodic forwardcount: error: checkpoints < 0")
    elseif periodic.steps < 1
        error("Periodic forwardcount: error: steps < 1")
    elseif periodic.bundle < 1
        error("Periodic forwardcount: error: bundle < 1")
    end
end