macro checkpoint(alg, adtool, loop)
    ex = quote
        function tobedifferentiated(inputs)
            local F_H = similar(inputs)
            local F = inputs
            $(loop.args[2])
            outputs = F
            return outputs
        end
        if isa($alg, Revolve)
            storemap = Dict{Int32,Int32}()
            check = 0
            F_Check = Array{Any, 2}(undef, 3, $alg.acp)
            F_final = Array{Float64, 1}(undef, 2)
            while true
                next_action = next_action!($alg)
                if (next_action.actionflag == Checkpointing.store)
                    check = check+1
                    storemap[next_action.iteration-1]=check
                    $alg.fstore(F,F_Check,t,check)
                elseif (next_action.actionflag == Checkpointing.forward)
                    for j= next_action.startiteration:(next_action.iteration - 1)
                        $(loop.args[2])
                    end
                elseif (next_action.actionflag == Checkpointing.firstuturn)
                    $(loop.args[2])
                    F_final .= F
                    L .= [0, 1]
                    t = 1.0-h
                    L_H .= L
                    L = Checkpointing.jacobian(tobedifferentiated, F_H, $adtool)[2,:]
                elseif (next_action.actionflag == Checkpointing.uturn)
                    L_H .= L
                    F_H = F
                    res = Checkpointing.jacobian(tobedifferentiated, F_H, $adtool)
                    L =  transpose(res)*L
                    t = t - h
                    if haskey(storemap,next_action.iteration-1-1)
                        delete!(storemap,next_action.iteration-1-1)
                        check=check-1
                    end
                elseif (next_action.actionflag == Checkpointing.restore)
                    F, t = $alg.frestore(F_Check,storemap[next_action.iteration-1])
                elseif next_action.actionflag == Checkpointing.done
                    if haskey(storemap,next_action.iteration-1-1)
                        delete!(storemap,next_action.iteration-1-1)
                        check=check-1
                    end
                    break
                end
            end
            F .= F_final
        elseif isa($alg, Periodic)
            check = 0
            F_Check = Array{Any, 2}(undef, 3, $alg.acp)
            F_final = Array{Float64, 1}(undef, 2)
            F_Check_inner = Array{Any, 2}(undef, 3, $alg.period)
            for i = 1:$alg.acp
                $alg.fstore(F,F_Check,t,i)
                for j= (i-1)*$alg.period: (i)*$alg.period-1
                    $(loop.args[2])
                end
            end
            F_final .= F
            L .= [0, 1]
            t = 1.0-h
            L_H .= L
            for i = $alg.acp:-1:1
                F,t = $alg.frestore(F_Check,i)
                for j= 1:$alg.period
                    $alg.fstore(F,F_Check_inner,t,j)
                    $(loop.args[2])
                end
                for j= $alg.period:-1:1
                    F,t = $alg.frestore(F_Check_inner,j)
                    L_H .= L
                    F_H .= F
                    res = Checkpointing.jacobian(tobedifferentiated, F_H, $adtool)
                    L =  transpose(res)*L
                    t = t - h
                end
            end
            F .= F_final
        elseif isa($alg, Online_r2)
            freeindices = Stack{Int32}()
            storemapinv = Dict{Int32,Int32}()
            storemap = Dict{Int32,Int32}()
            check = 0
            F_Check = Array{Any, 2}(undef, 3, $alg.acp)
            F_final = Array{Float64, 1}(undef, 2)
            oldcapo=0
            onlinesteps=0
            while $(loop.args[1])
                next_action = next_action!($alg)
                if (next_action.actionflag == Checkpointing.store)
                    check=next_action.cpnum+1
                    storemapinv[check]=next_action.iteration
                    $alg.fstore(F,F_Check,t,check)
                elseif (next_action.actionflag == Checkpointing.forward)
                    for j= oldcapo:(next_action.iteration-1)
                        if($(loop.args[1]))
                            $(loop.args[2])
                        end
                        onlinesteps=onlinesteps+1
                    end
                    oldcapo=next_action.iteration
                else
                    @error("Unexpected action in online phase: ", next_action.actionflag)
                end
            end
            for (key, value) in storemapinv
                storemap[value]=key
            end

            #Switch to offline revolve now.
            update_revolve($alg, onlinesteps+1)
            while true
                next_action = next_action!($alg.revolve)
                if (next_action.actionflag == Checkpointing.store)
                    check=pop!(freeindices)
                    storemap[next_action.iteration-1]=check
                    $alg.revolve.fstore(F,F_Check,t,check)
                elseif (next_action.actionflag == Checkpointing.forward)
                    for j= next_action.startiteration:(next_action.iteration-1)
                        $(loop.args[2])
                    end
                elseif (next_action.actionflag == Checkpointing.firstuturn)
                    $(loop.args[2])
                    F_final .= F
                    L .= [0, 1]
                    t = 1.0-h
                    L_H .= L
                    L = Checkpointing.jacobian(tobedifferentiated, F_H, $adtool)[2,:]
                elseif (next_action.actionflag == Checkpointing.uturn)
                    L_H .= L
                    F_H = F
                    res = Checkpointing.jacobian(tobedifferentiated, F_H, $adtool)
                    L =  transpose(res)*L
                    t = t - h
                    if haskey(storemap,next_action.iteration-1-1)
                        push!(freeindices, storemap[next_action.iteration-1-1])
                        delete!(storemap,next_action.iteration-1-1)
                    end
                elseif (next_action.actionflag == Checkpointing.restore)
                    F, t = $alg.revolve.frestore(F_Check,storemap[next_action.iteration-1])
                elseif next_action.actionflag == Checkpointing.done
                    if haskey(storemap,next_action.iteration-1-1)
                        delete!(storemap,next_action.iteration-1-1)
                    end
                    break
                end
            end
            F .= F_final
        end
    end
    esc(ex)
end
