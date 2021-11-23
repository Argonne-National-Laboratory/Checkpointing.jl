# This provides the functionality of periodic checkpointing. It uses the
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
    storedorrestored::Bool
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
    curriter = 0
    tail   = 0
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
    storedorrestored = false

    periodic = Periodic(steps, bundle, tail, acp, curriter, cend, numfwd, numinv, numstore, rwcp, prevcend, period, firstuturned, stepof, storedorrestored, verbose, fstore, frestore)

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
    if (!periodic.firstuturned)
        if (periodic.curriter == (periodic.acp-1)*periodic.period)
                actionflag = firstuturn
                periodic.firstuturned = true
                periodic.storedorrestored = false
                startiteration = periodic.curriter
                iteration = (periodic.curriter + periodic.period) + periodic.tail

        else
            if (periodic.storedorrestored)
                actionflag = forward
                periodic.storedorrestored = false
                startiteration = periodic.curriter
                iteration = (periodic.curriter + periodic.period)
                periodic.curriter = periodic.curriter + periodic.period
            else
                actionflag = store
                periodic.storedorrestored = true
                iteration = periodic.curriter 
            end
        end
    else
        if (periodic.curriter == 0) && (actionflag == uturn)
            actionflag = done
        else
            if (periodic.storedorrestored)
                actionflag = uturn
                periodic.storedorrestored = false
                startiteration = periodic.curriter
                iteration = (periodic.curriter + periodic.period)
            else
                periodic.curriter = periodic.curriter - periodic.period
                if (periodic.curriter < 0)
                    actionflag = done
                else
                    actionflag = restore
                    periodic.storedorrestored = true
                    iteration = periodic.curriter
                end
            end
        end
    end

    if periodic.verbose > 1
        if actionflag == forward
            @info " run forward iterations    [$startiteration, $(iteration - 1)]"
        elseif actionflag == restore
            @info " restore input of iteration $iteration"
        elseif actionflag == firstuturn
            @info " 1st uturn for iterations  [$startiteration, $(iteration - 1)]"
        elseif actionflag == uturn
            @info " uturn for iterations      [$startiteration, $(iteration - 1)]"
        elseif actionflag == store
            @info " store input of iteration $iteration  "
        elseif actionflag == done
            @info " done"
        end
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