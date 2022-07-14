# This provides the functionality of online checkpointing.
# It is based on an implementation of the Online_r2 algorithm from:
#   Philipp Stumm and Andrea Walther. 2010. New Algorithms for
#   Optimal Online Checkpointing. SIAM J. Sci. Comput. 32, 2
#   (March 2010), 836–854. https://doi.org/10.1137/080742439
# It is, furthermore, a Julia translation of the original C++
# code distributed by the PyRevolve project:
#   https://github.com/devitocodes/pyrevolve/blob/master/src/revolve.cpp
# TODO: Extend Online_r2 to Online_r3

mutable struct Online_r2{MT} <: Scheme where {MT}
	check::Int
	capo::Int
    acp::Int
    numfwd::Int
    numcmd::Int
    numstore::Int
    oldcapo::Int
    ind::Int
    oldind::Int
    iter::Int
    incr::Int
    offset::Int
    t::Int
    verbose::Bool
    fstore::Union{Function,Nothing}
    frestore::Union{Function,Nothing}
    ch::Vector{Int}
    ord_ch::Vector{Int}
    num_rep::Vector{Int}
    revolve::Revolve
end

function Online_r2{MT}(
    checkpoints::Int,
    fstore::Union{Function,Nothing} = nothing,
    frestore::Union{Function,Nothing} = nothing,
    anActionInstance::Union{Nothing,Action} = nothing,
    verbose::Int = 0
) where {MT}
    if !isa(anActionInstance, Nothing)
        anActionInstance.actionflag = 0
        anActionInstance.iteration  = 0
        anActionInstance.cpNum      = 0
    end
    if checkpoints < 0
       @error("Online_r2: negative checkpoints")
    end
    acp             = checkpoints
    numfwd          = 0
    numcmd          = 0
    numstore        = 0
    oldcapo         = 0
    check = -1
    capo = 0
    oldind = -1
    ind = -1
    iter = -1
    incr = -1
    offset= -1
    t=-1
    ch = Vector{Int}(undef, acp)
    ord_ch = Vector{Int}(undef, acp)
    num_rep = Vector{Int}(undef, acp)
    for i in 1:acp
        ch[i] = -1
        ord_ch[i] = -1
        num_rep[i] = -1
    end
    verbose = false
    revolve = Revolve{MT}(typemax(Int64), acp, fstore, frestore; verbose=3)
    online_r2 = Online_r2{MT}(check, capo, acp, numfwd, numcmd, numstore,
                            oldcapo, ind, oldind, iter, incr, offset, t,
                            verbose, fstore, frestore, ch, ord_ch, num_rep, revolve)
    return online_r2
end

function update_revolve(online::Online_r2{MT}, steps) where {MT}
    online.revolve = Revolve{MT}(steps, online.acp, online.fstore, online.frestore)
    online.revolve.rwcp = online.revolve.acp-1
    online.revolve.steps = steps
    online.revolve.acp = online.acp
    online.revolve.cstart =  steps-1
    online.revolve.cend = steps
    online.revolve.numfwd = steps-1
    online.revolve.numinv= online.revolve.numfwd-1
    online.revolve.numstore= online.acp
    online.revolve.prevcend= steps
    online.revolve.firstuturned=false
    online.revolve.verbose= 0
    num_ch = Vector{Int}(undef, online.acp)
    for i=1:online.acp
        num_ch[i] = 1
        for j=1:online.acp
            if (online.ch[j] < online.ch[i])
                num_ch[i] = num_ch[i]+1
            end
        end
    end
    for i=1:online.acp
        for j=1:online.acp
            if (num_ch[j] == i)
                online.ord_ch[i]=j;
            end
        end
    end
    for j=1:online.acp
        online.revolve.stepof[j] = online.ch[online.ord_ch[j]]
    end
    online.revolve.stepof[online.acp+1]=0
end

function next_action!(online::Online_r2)::Action
    # Default values for next action
    actionflag     = none
    if online.verbose
        if(online.check !=-1)
            @info(online.check+1,  online.ch[online.check+1],  online.capo)
            for i in 1:online.acp
                println("online.ch[",i,"] =", online.ch[i])
            end
        else
            @info(online.check, online.capo)
            for i in 1:online.acp
                println("online.ch[",i,"] =", online.ch[i])
            end
        end
    end
    online.numcmd+=1
    #We use this logic because the C++ version uses short circuiting
    cond2 = false
    if online.check != -1
      cond2 = online.ch[online.check+1] != online.capo
    end
    online.oldcapo = online.capo
    if ((online.check == -1) || ( cond2 && (online.capo <= online.acp-1)))
    #condition for takeshot for r=1
    #   (If no checkpoint has been taken before OR
    #    If a store has not just occurred AND the iteration count is
    #    less than the total number of checkpoints)
        if online.verbose
            @info("condition for takeshot for r=1")
        end
        online.check += 1
        online.ch[online.check+1] = online.capo
        online.t = 0
        if (online.acp < 4)
            for i in 1:online.acp
      	        online.num_rep[i] = 2
            end
            online.incr = 2
            online.iter = 1
            online.oldind = online.acp-1
        else
            online.iter = 1
            online.incr = 1
            online.oldind = 1
            for i in 1:online.acp
      	        online.num_rep[i] = 1
      	        online.ord_ch[i] = i-1
            end
            online.offset = online.acp-1
        end
        if (online.capo == online.acp-1)
            online.ind = 2
        end
        # Increase the number of takeshots and the corresponding checkpoint
        online.numstore+=1
        return Action(store, online.capo-1, -1, online.check)
    elseif (online.capo < online.acp-1)
    #condition for advance for r=1
    #   (the iteraton is less that the total number of checkpoints)
        if online.verbose
            @info("condition for advance for r=1")
        end
        online.capo = online.oldcapo+1
        online.numfwd+=1
        return Action(forward, online.capo, online.oldcapo, -1)
    else
    #Online_r2-Checkpointing for r=2
        if (online.ch[online.check+1] == online.capo)
            # condition for advance for r=2
            # (checkpoint has just occurred)
            if online.verbose
                @info("Online_r2-condition for advance for r=2 online.acp=", online.acp)
            end
            if (online.acp == 1)
                online.capo = BigInt(typemax(Int64))
      		    online.numfwd+=1
                return Action(forward, online.capo, online.oldcapo, -1)
            elseif (online.acp == 2)
                online.capo = online.ch[1+1]+online.incr
      		    online.numfwd+=1
                return Action(forward, online.capo, online.oldcapo, -1)
            elseif (online.acp == 3)
                online.numfwd+=online.incr
      		    if (online.iter == 0)
      			    online.capo = online.ch[online.oldind+1]
                    for i=0:(online.t+1)/2
      				    online.capo += online.incr
      				    online.incr = online.incr + 1
      				    online.iter = online.iter + 1
                    end
      		    else
      			    online.capo = online.ch[online.ind+1]+online.incr
      			    online.incr = online.incr + 1
      				online.iter = online.iter + 1
                end
                actionflag = forward
                return Action(forward, online.capo, online.oldcapo, -1)
      	    else
                if online.verbose
                     @info("Online_r2-condition for advance for r=2 online.acp-1=", online.acp-1," online.capo= ", online.capo)
                end
                if (online.capo == online.acp-1)
      			    online.capo = online.capo+2
      		    	online.ind=online.acp-1
      			    online.numfwd+=2
                    return Action(forward, online.capo, online.oldcapo, -1)
                end
      		    if (online.t == 0)
      			    if (online.iter < online.offset)
      				    online.capo = online.capo+1
      				    online.numfwd+=1
      			    else
      				    online.capo = online.capo+2
      				    online.numfwd+=2
                    end
      			    if (online.offset == 1)
      				    online.t += 1
                    end
                    return Action(forward, online.capo, online.oldcapo, -1)
                end
      		    if (online.verbose)
                    @info(" iter ", iter, "incr ", incr)
                end
                error(" not implemented yet")
                return Action(done, online.capo, online.oldcapo, -1)
            end
        else
            #takeshot for r=2
            if (online.verbose)
                @info("Online_r2-condition for takeshot for r=2 online.acp =", online.acp)
            end
            if (online.acp == 2)
                online.ch[1+1] = online.capo
                online.incr+=1
      		    #Increase the number of takeshots and the corresponding checkpoint
      		    online.numstore+=1
                return Action(store, online.capo-1, -1, 1+1)
            elseif (online.acp == 3)
                online.ch[online.ind+1] = online.capo
      		    online.check = online.ind
                if (online.verbose)
      		        @info(" iter ", online.iter, " online.num_rep[1] ", online.num_rep[1+1])
                end
      		    if (online.iter == online.num_rep[1+1])
                    online.iter = 0
      			    online.t+=1
      		    	online.oldind = online.ind
      			    online.num_rep[1+1]+=1
      			    online.ind = 2 - online.num_rep[1+1]%2
      			    online.incr=1
                end
      		    #Increase the number of takeshots and the corresponding checkpoint
      		    online.numstore+=1
                return Action(store, online.capo-1, -1, online.check)
            else
                if (online.verbose)
                    @info(" online.capo ", online.capo, " online.acp ", online.acp)
                end
                if (online.capo < online.acp+2)
      			    online.ch[online.ind+1] = online.capo
      			    online.check = online.ind
      			    if (online.capo == online.acp+1)
                        online.oldind = online.ord_ch[online.acp-1+1]
                        online.ind = online.ch[online.ord_ch[online.acp-1+1]+1]
      				    if (online.verbose)
                            @info(" oldind ", online.oldind, " ind ", online.ind)
                        end
                        for k=online.acp:-1:3
      					    online.ord_ch[k]=online.ord_ch[k-1]
      					    online.ch[online.ord_ch[k]+1] = online.ch[online.ord_ch[k-1]+1]
                        end
      				    online.ord_ch[1+1] = online.oldind
      				    online.ch[online.ord_ch[1+1]+1] = online.ind
      				    online.incr = 2
      				    online.ind = 2
      				    if (online.verbose)
      				    	@info(" ind ", online.ind, " incr ", online.incr, " iter ", online.iter)
                            for j=1:online.acp
      				    	    @info(" j ", j, " ord_ch ", online.ord_ch[j], " ch ", online.ch[online.ord_ch[j]+1], " rep ", online.num_rep[online.ord_ch[j]+1])
                            end
                        end
                    end
      			    #Increase the number of takeshots and the corresponding checkpoint
      			    online.numstore+=1
                    return Action(store, online.capo-1, -1, online.check)
                end

                if (online.t == 0)
                    if (online.verbose)
                        @info(" online.ind ", online.ind, " online.incr ",  online.incr, " iter ", online.iter, " offset ", online.offset)
                    end
                    if (online.iter == online.offset)
                        online.offset=online.offset-1
                        online.iter = 1
                        online.check = online.ord_ch[online.acp-1+1]
                        online.ch[online.ord_ch[online.acp-1+1]+1] = online.capo
                        online.oldind = online.ord_ch[online.acp-1+1]
                        online.ind = online.ch[online.ord_ch[online.acp-1+1]+1]
                        if (online.verbose)
                            @info(" oldind " , online.oldind , " ind " , online.ind)
                        end
                        for k=online.acp-1:-1:online.incr+1
                            online.ord_ch[k+1]=online.ord_ch[k-1+1]
                            online.ch[online.ord_ch[k+1]+1] = online.ch[online.ord_ch[k-1+1]+1]
                        end
                        online.ord_ch[online.incr+1] = online.oldind
                        online.ch[online.ord_ch[online.incr+1]+1] = online.ind
                        online.incr+=1
                        online.ind=online.incr
                        if (online.verbose)
                            @info(" ind ", online.ind, " incr ", online.incr, " iter ", online.iter)
                            for j=1:online.acp
                                @info(" j ", j, " ord_ch ", online.ord_ch[j], " ch ", online.ch[online.ord_ch[j]+1], " rep ", online.num_rep[online.ord_ch[j]+1])
                            end
                        end
                    else
                        online.ch[online.ord_ch[online.ind+1]+1] = online.capo
                        online.check = online.ord_ch[online.ind+1]
                        online.iter+=1
                        online.ind+=1
                        if (online.verbose)
                            @info(" xx ind ", online.ind, " incr ", online.incr, " iter ", online.iter)
                        end
                    end
                    #Increase the number of takeshots and the corresponding checkpoint
                    online.numstore=online.numstore+1
                    return Action(store, online.capo-1, -1, online.check)
                end

            end
        end
    end
    # This means that the end of Online_r2 Checkpointing for r=2 is reached and
    #  another Online_r2 Checkpointing class must be started
    @info("Online_r2 is optimal over the range [0,(numcheckpoints+2)*(numcheckpoints+1)/2]. Online_r3 needs to be implemented")
    return Action(error, online.capo, online.oldcapo, -1)
end
