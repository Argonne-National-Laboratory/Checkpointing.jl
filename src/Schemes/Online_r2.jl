# TODO Add Explanation

mutable struct Online_r2 <: Scheme
    #check is last stored checkpoint
	check::Int
	#capo is the temporary fine
	capo::Int
    tail::Int
    acp::Int
    cstart::Int
    cend::Int
    numfwd::Int
    numcmd::Int
    numstore::Int
    rwcp::Int
    prevcend::Int
    firstuturned::Bool
    verbose::Int
    fstore::Union{Function,Nothing}
    frestore::Union{Function,Nothing}
    ch::Vector{Int}
    ord_ch::Vector{Int}
    num_rep::Vector{Int}
end

function Online_r2(
    #steps::Int,
    checkpoints::Int,
    fstore::Union{Function,Nothing} = nothing,
    frestore::Union{Function,Nothing} = nothing,
    anActionInstance::Union{Nothing,Action} = nothing,
    #bundle_::Union{Nothing,Int} = nothing,
    verbose::Int = 0
)
    if !isa(anActionInstance, Nothing)
        # same as default init above
        anActionInstance.actionflag = 0
        anActionInstance.iteration  = 0
        anActionInstance.cpNum      = 0
    end
    if steps<0
       error("Online_r2: negative steps")
    elseif checkpoints < 0
       error("Online_r2: negative checkpoints")
    end
    cstart = 0
    tail   = 1
    cend            = steps
    acp             = checkpoints
    numfwd          = 0
    numcmd          = 0
    numstore        = 0
    rwcp            = -1
    prevcend        = 0
    firstuturned    = false
    check = -1
    capo = -1
    ch = Vector{Int}(undef, acp)
    ord_ch = Vector{Int}(undef, acp)
    num_rep = Vector{Int}(undef, acp)
    revolve = Online_r2(check, capo, tail, acp, cstart, cend, numfwd, numcmd, numstore, rwcp, prevcend, firstuturned, verbose, fstore, frestore, ch, ord_ch, num_rep)
    return revolve
end

function next_action!(revolve::Online_r2)::Action
    # Default values for next action
    actionflag     = none
    info(" check = ", check,  " ch[check] ", ch[check], " capo ", capo) 
    numcmd+=1
    cpnum = 0

    if ((check == -1) || ((ch[check] != capo) && (capo <= snaps-1)))
    #condition for takeshot for r=1
        oldcapo_o = capo
        check += 1
        ch[check] = capo
        t = 0
        if (snaps < 4)
            for i in 0:snaps-1
            #for(int i=0;i<snaps;i++)
      	        num_rep[i] = 2
            end
            incr = 2
            iter = 1
            oldind = snaps-1
        else
            iter = 1
            incr = 1
            oldind = 1
            for i in 0:snaps-1
            #for(int i=0;i<snaps;i++)      
      	        num_rep[i] = 1
      	        ord_ch[i] = i
            end
            offset = snaps-1
        end
        if (capo == snaps-1)
            ind = 2
            old_f = 1
        end    
        # Increase the number of takeshots and the corresponding checkpoint
        numstore+=1
        #checkpoint->number_of_writes[check]++
        return Action(store, check, -1, -1)
        #return ACTION::store
    elseif (capo < snaps-1)
    #condition for advance for r=1
        capo = oldcapo_o+1
        nunmfwd+=1
        return Action(forward, capo, oldcapo_o, cpnum)
        #return ACTION::advance
    else
    #Online_r2-Checkpointing for r=2
        if (ch[check] == capo)
        # condition for advance for r=2
            if (snaps == 1)
                capo = MAXINT-1
      		    nunmfwd+=1
                return Action(forward, capo, oldcapo_o, cpnum)
      		    #return ACTION::advance
            elseif (snaps == 2)
                capo = ch[1]+incr
      		    nunmfwd+=1
                return Action(forward, capo, oldcapo_o, cpnum)
      		    #return ACTION::advance
            elseif (snaps == 3) 
                nunmfwd+=incr
      		    if (iter == 0)
      			    capo = ch[oldind]
                    for i=0:(t+1)/2
      			        #for(int i=0;i<=(t+1)/2;i++)
      				    capo += incr
      				    incr = incr + 1
      				    iter = iter + 1
                    end
      		    else
      			    capo = ch[ind]+incr
      			    incr = incr + 1
      				iter = iter + 1
                end
                actionflag = forward
                return Action(forward, capo, oldcapo_o, cpnum)
      		    #return ACTION::advance
      	    else
                if (capo == snaps-1)
      			    capo = capo+2
      		    	ind=snaps-1
      			    nunmfwd+=2
                    return Action(forward, capo, oldcapo_o, cpnum)
      			    #return ACTION::advance
                end
      		    if (output)
      			    info(" iter ", iter,  " incr ", incr, "offset", offset)
                end
      		    if (t == 0)
      			    if (iter < offset)
      				    capo = capo+1
      				    nunmfwd+=1
      			    else
      				    capo = capo+2
      				    nunmfwd+=2
                    end
      			    if (offset == 1)
      				    t+=1
                    end
                    return Action(forward, capo, oldcapo_o, cpnum)
      			    #return ACTION::advance
                end
      		    if (output)
                    info(" iter ", iter, "incr ", incr)
                end
                error(" not implemented yet")
                return Action(done, capo, oldcapo_o, cpnum)
      		    #return ACTION::error
            end
        else
            #takeshot for r=2
            if (snaps == 2)
                ch[1] = capo
      		    incr+=1
      		    #Increase the number of takeshots and the corresponding checkpoint
      		    numstore+=1
      		    #checkpoint->number_of_writes[1]++
                return Action(store, check, -1, -1)
      		    #return ACTION::takeshot
            elseif (snaps == 3) 
                ch[ind] = capo
      		    check = ind
      		    info(" iter ", iter, " num_rep[1] ", num_rep[1])
      		    if (iter == num_rep[1])
      			    iter = 0
      			    t+=1
      		    	oldind = ind
      			    num_rep[1]+=1
      			    ind = 2 - num_rep[1]%2
      			    incr=1
                  end
      		    #Increase the number of takeshots and the corresponding checkpoint
      		    numstore+=1
      		    #checkpoint->number_of_writes[check]++
                return Action(store, check, -1, -1)
      		    #return ACTION::takeshot
            else 
                if (capo < snaps+2)
      			    ch[ind] = capo
      			    check = ind
      			    if (capo == snaps+1)
      				    oldind = ord_ch[snaps-1]
      				    ind = ch[ord_ch[snaps-1]]
      				    if (output)
                            info(" oldind ", oldind, " ind ", ind)
                        end
                        for k=snaps-1:-1:2
      				    #for(int k=snaps-1;k>1;k--)
      					    ord_ch[k]=ord_ch[k-1]
      					    ch[ord_ch[k]] = ch[ord_ch[k-1]]
                        end
      				    ord_ch[1] = oldind
      				    ch[ord_ch[1]] = ind
      				    incr=2
      				    ind = 2
      				    if (output)
      				    	info(" ind ", ind, " incr ", incr, " iter ", iter)
                            for j=0:snaps-1
      				    	#for(int j=0;j<snaps;j++)
      				    	    info(" j ", j, " ord_ch ", ord_ch[j], " ch ", ch[ord_ch[j]], " rep ", num_rep[ord_ch[j]])
                            end
                        end
                    end
      			    #Increase the number of takeshots and the corresponding checkpoint
      			    numstore+=1
      			    #checkpoint->number_of_writes[check]++
                    return Action(store, check, -1, -1)
      			    #return ACTION::takeshot
                end
            end
      		if (t == 0)
      			if (output)
                    info(" ind ", ind, " incr ",  incr, " iter ", iter, " offset ", offset)
                end
                if (iter == offset)
      				offset=offset-1
      				iter = 1
      				check = ord_ch[snaps-1]
      				ch[ord_ch[snaps-1]] = capo
      				oldind = ord_ch[snaps-1]
      				ind = ch[ord_ch[snaps-1]]
      				if (output)
                        info(" oldind " , oldind , " ind " , ind)
                    end
                    for k=snaps-1:-1:incr+1
      				#for(int k=snaps-1;k>incr;k--)
      					ord_ch[k]=ord_ch[k-1]
      					ch[ord_ch[k]] = ch[ord_ch[k-1]]
                    end
      				ord_ch[incr] = oldind
      				ch[ord_ch[incr]] = ind
      				incr+=1
      				ind=incr
      				if (output)
                        info(" ind ", ind, " incr ", incr, " iter ", iter)
                        for j=0:snaps-1
      					#for(int j=0;j<snaps;j++)
                            info(" j ", j << " ord_ch ", ord_ch[j], " ch ", ch[ord_ch[j]], " rep ", num_rep[ord_ch[j]])
                        end
                    end
      			else
      				ch[ord_ch[ind]] = capo
      				check = ord_ch[ind]
      				iter+=1
      				ind+=1
      				if (output)
                        info(" xx ind ", ind, " incr ", incr, " iter ", iter)
                    end
                end
      			#Increase the number of takeshots and the corresponding checkpoint
      			numstore=numstore+1
      			#checkpoint->number_of_writes[check]++
                return Action(store, check, -1, -1)
      			#return ACTION::takeshot
      		end
        end
    end
    actionflag = done
    return Action(done, capo, oldcapo_o, cpnum)
    #return ACTION::terminate # This means that the end of Online_r2 Checkpointing for r=2 is reached and
    #  another Online_r2 Checkpointing class must be started
end


function Revolve(
    online::Online_r2,
    checkpoints::Int,
    fstore::Union{Function,Nothing} = nothing,
    frestore::Union{Function,Nothing} = nothing;
)
    if !isa(anActionInstance, Nothing)
        # same as default init above
        anActionInstance.actionflag = 0
        anActionInstance.iteration  = 0
        anActionInstance.cpNum      = 0
    end
    cstart = online_r2.cstart
    tail   = 1
    cend            = online_r2.steps
    acp             = online_r2.checkpoints
    numfwd          = online_r2.numfwd
    numinv          = online_r2.numinv
    numstore        = online_r2.numstore
    rwcp            = online_r2.rwcp
    prevcend        = online_r2.prevcend
    firstuturned    = online_r2.firstuturned #true
    stepof = Vector{Int}(undef, acp+1)

    revolve = Revolve(steps, bundle, tail, acp, cstart, cend, numfwd, numinv, numstore, rwcp, prevcend, firstuturned, stepof, verbose, fstore, frestore)

    if verbose > 0
        predfwdcnt = forwardcount(revolve)
        if predfwdcnt == -1
            error("Revolve: error returned by  revolve::forwardcount")
        else
            @info "prediction:"
            @info " overhead forward steps : $predfwdcnt"
            @info " overhead factor        : $(predfwdcnt/steps)"
        end
    end
    return revolve
end

function checkpoint_mutable(body::Function, alg::Online_r2, model_input::MT, shadowmodel::MT) where {MT}
    model = deepcopy(model_input)
    storemap = Dict{Int32,Int32}()
    check = 0
    model_check = Array{MT}(undef, alg.acp)
    copyto!(model_input, deepcopy(model))
    while true
        next_action = next_action!(alg)
        if (next_action.actionflag == Checkpointing.store)
            check = check+1
            storemap[next_action.iteration-1]=check
            model_check[check] = deepcopy(model)
        elseif (next_action.actionflag == Checkpointing.forward)
            for j= next_action.startiteration:(next_action.iteration - 1)
                body(model)
            end
        elseif (next_action.actionflag == Checkpointing.firstuturn)
            error("Unexpected firstuturn")
        elseif (next_action.actionflag == Checkpointing.uturn)
            error("Unexpected uturn")
        elseif (next_action.actionflag == Checkpointing.restore)
            error("Unexpected restore")
        elseif next_action.actionflag == Checkpointing.done
            info("Done with online phase")
            break
        end
    end
end
