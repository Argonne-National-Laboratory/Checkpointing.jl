module Checkpointing

using LinearAlgebra

export mynorm

abstract type Scheme end
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

struct Action
	actionflag::ActionFlag
	iteration::Int
	startiteration::Int
	cpnum::Int
end

abstract type AbstractADTool end

function jacobian(tobedifferentiated, F_H, ::AbstractADTool)
    error("No AD tool interface implemented")
end

export AbstractADTool, jacobian, @checkpoint, @checkpoint_mutable

include("Schemes/Revolve.jl")
include("Schemes/Periodic.jl")

export Revolve, guess, factor, next_action!, ActionFlag, Periodic
export ReverseDiffADTool, ZygoteADTool, EnzymeADTool, ForwardDiffADTool, DiffractorADTool, jacobian

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
        end
    end
    esc(ex)
end

macro checkpoint_mutable(alg, adtool, model, shadowmodel, loop)
    ex = quote
        function tobedifferentiated($model)
            $(loop.args[2])
            return nothing
        end
        if isa($alg, Revolve)
            storemap = Dict{Int32,Int32}()
            check = 0
            MT = typeof($model)
            model_check = Array{MT}(undef, $alg.acp)
            model_final = deepcopy($model)
            while true
                next_action = next_action!($alg)
                if (next_action.actionflag == Checkpointing.store)
                    check = check+1
                    storemap[next_action.iteration-1]=check
                    model_check[check] = deepcopy($model)
                elseif (next_action.actionflag == Checkpointing.forward)
                    for j= next_action.startiteration:(next_action.iteration - 1)
                        $(loop.args[2])
                    end
                elseif (next_action.actionflag == Checkpointing.firstuturn)
                    $(loop.args[2])
                    model_final = deepcopy($model)
                    Enzyme.autodiff(tobedifferentiated, Duplicated($model,$shadowmodel))
                elseif (next_action.actionflag == Checkpointing.uturn)
                    Enzyme.autodiff(tobedifferentiated, Duplicated($model,$shadowmodel))
                    if haskey(storemap,next_action.iteration-1-1)
                        delete!(storemap,next_action.iteration-1-1)
                        check=check-1
                    end
                elseif (next_action.actionflag == Checkpointing.restore)
                    $model = deepcopy(model_check[storemap[next_action.iteration-1]])
                elseif next_action.actionflag == Checkpointing.done
                    if haskey(storemap,next_action.iteration-1-1)
                        delete!(storemap,next_action.iteration-1-1)
                        check=check-1
                    end
                    break
                end
            end
            $model = deepcopy(model_final)
        elseif isa($alg, Periodic)
            MT = typeof($model)
            model_check_outer = Array{MT}(undef, $alg.acp)
            model_check_inner = Array{MT}(undef, $alg.period)
            model_final = deepcopy($model)
            check = 0
            for i = 1:$alg.acp
                model_check_outer[i] = deepcopy($model)
                for j= (i-1)*$alg.period: (i)*$alg.period-1
                    $(loop.args[2])
                end
            end
            model_final = deepcopy($model)
            for i = $alg.acp:-1:1
                $model = deepcopy(model_check_outer[i])
                for j= 1:$alg.period
                    model_check_inner[j] = deepcopy($model)
                    $(loop.args[2])
                end
                for j= $alg.period:-1:1
                    $model = deepcopy(model_check_inner[j])
                    Enzyme.autodiff(tobedifferentiated, Duplicated($model,$shadowmodel))
                end
            end
            $model = deepcopy(model_final)
        end
    end
    esc(ex)
end

end
