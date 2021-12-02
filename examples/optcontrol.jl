# This is a Julia version of Solution of the optimal control problem
# based on code written by Andrea Walther. See:
# Walther, Andrea, and Narayanan, Sri Hari Krishna. Extending the Binomial Checkpointing
# Technique for Resilience. United States: N. p., 2016. https://www.osti.gov/biblio/1364654.

using Checkpointing
using Enzyme

function func_U(t)
    e = exp(1)
    return 2.0*((e^(3.0*t))-(e^3))/((e^(3.0*t/2.0))*(2.0+(e^3)))
end

function func(X,t)
    F = similar(X)
    F[1] = 0.5*X[1]+ func_U(t)
    F[2] = X[1]*X[1]+0.5*(func_U(t)*func_U(t))
    return F
end

function func_adj(BF, X)
    BX = similar(X)
    BX[1] = 0.5*BF[1]+2.0*X[1]*BF[2]
    BX[2] = 0.0
    return BX
end

function opt_sol(Y,t)
    e = exp(1)
    Y[1] = (2.0*e^(3.0*t)+e^3)/(e^(3.0*t/2.0)*(2.0+e^3))
    Y[2] = (2.0*e^(3.0*t)-e^(6.0-3.0*t)-2.0+e^6)/((2.0+e^3)^2)
    return
end

function opt_lambda(L,t)
    e = exp(1)
    L[1] = (2.0*e^(3-t)-2.0*e^(2.0*t))/(e^(t/2.0)*(2+e^3))
    L[2] = 1.0
    return
end

function advance(F,F_H,t,h)
    k0 = func(F_H,t)
    G = similar(F_H)
    G[1] = F_H[1] + h/2.0*k0[1]
    G[2] = F_H[2] + h/2.0*k0[2]
    k1 = func(G,t+h/2.0)
    F[1] = F_H[1] + h*k1[1]
    F[2] = F_H[2] + h*k1[2]
    return
end

function adjoint(F_H,L_H,F,L,t,h)
    k0 = func(F_H,t)
    G = similar(F_H)
    Bk1 = similar(L_H)
    G[1] = F_H[1] + h/2.0*k0[1]
    G[2] = F_H[2] + h/2.0*k0[2]
    k1 = func(G,t+h/2.0)
    L[1] = L_H[1]
    L[2] = L_H[2]
    Bk1[1] = h*L_H[1]
    Bk1[2] = h*L_H[2]
    BG = func_adj(Bk1,G)
    Bk0 = similar(BG)
    L[1] += BG[1]
    L[2] += BG[2]
    Bk0[1] = h/2.0*BG[1]
    Bk0[2] = h/2.0*BG[2]
    BH = func_adj(Bk0,F_H)
    L[1] += BH[1]
    L[2] += BH[2]
    return
end


function header()
        println("**************************************************************************")
        println("*              Solution of the optimal control problem                   *")
        println("*                                                                        *")
        println("*                     J(y) = y_2(1) -> min                               *")
        println("*           s.t.   dy_1/dt = 0.5*y_1(t) + u(t),            y_1(0)=1      *")
        println("*                  dy_2/dt = y_1(t)^2 + 0.5*u(t)^2         y_2(0)=0      *")
        println("*                                                                        *")
        println("*                  the adjoints equations fulfill                        *")
        println("*                                                                        *")
        println("*         dl_1/dt = -0.5*l_1(t) - 2*y_1(t)*l_2(t)          l_1(1)=0      *")
        println("*         dl_2/dt = 0                                      l_2(1)=1      *")
        println("*                                                                        *")
        println("*   with Revolve for Online and (Multi-Stage) Offline Checkpointing      *")
        println("*                                                                        *")
        println("**************************************************************************")

        println("**************************************************************************")
        println("*        The solution of the optimal control problem above is            *")
        println("*                                                                        *")
        println("*        y_1*(t) = (2*e^(3t)+e^3)/(e^(3t/2)*(2+e^3))                     *")
        println("*        y_2*(t) = (2*e^(3t)-e^(6-3t)-2+e^6)/((2+e^3)^2)                 *")
        println("*          u*(t) = (2*e^(3t)-e^3)/(e^(3t/2)*(2+e^3))                     *")
        println("*        l_1*(t) = (2*e^(3-t)-2*e^(2t))/(e^(t/2)*(2+e^3))                *")
        println("*        l_2*(t) = 1                                                     *")
        println("*                                                                        *")
        println("**************************************************************************")

        return
end

macro checkpoint(alg, forloop)
    ex = quote        
        if(isa($alg, Revolve))
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
                        $(forloop.args[2])
                    end
                elseif (next_action.actionflag == Checkpointing.firstuturn)
                    $(forloop.args[2])
                    F_final .= F
                    L .= [0, 1]
                    t = 1.0-h
                    L_H .= L
                    function tobedifferentiated(F,F_H)
                        $(forloop.args[2])
                        return nothing
                    end
                    # @show typeof(F)
                    # @show length(F)
                    # @show typeof(L)
                    # @show length(L)
                    # @show typeof(F_H)
                    # @show length(F_H)
                    # @show typeof(L_H)
                    # @show length(L_H)
                    # autodiff(tobedifferentiated, Duplicated(F, L), Duplicated(F_H, L_H))
                    adjoint(F_H ,L_H, F, L, t, h)
                elseif (next_action.actionflag == Checkpointing.uturn)
                    L_H .= L
                    F_H .= F
                    adjoint(F_H, L_H, F, L, t, h)
                    t = t - h
                    if haskey(storemap,next_action.iteration-1-1)
                        delete!(storemap,next_action.iteration-1-1)
                        check=check-1
                    end
                elseif (next_action.actionflag == Checkpointing.restore)
                    t = $alg.frestore(F,F_Check,storemap[next_action.iteration-1])
                elseif next_action.actionflag == Checkpointing.done
                    if haskey(storemap,next_action.iteration-1-1)
                        delete!(storemap,next_action.iteration-1-1)
                        check=check-1
                    end
                    break
                end
            end
            F .= F_final
        elseif(isa($alg, Periodic))
            check = 0
            F_Check = Array{Any, 2}(undef, 3, $alg.acp)
            F_final = Array{Float64, 1}(undef, 2)
            F_Check_inner = Array{Any, 2}(undef, 3, $alg.period)
            for i = 1:$alg.acp
                $alg.fstore(F,F_Check,t,i)
                for j= (i-1)*$alg.period: (i)*$alg.period-1
                    $(forloop.args[2])
                end
            end
            F_final .= F
            L .= [0, 1]
            t = 1.0-h
            L_H .= L
            for i = $alg.acp:-1:1
                t = $alg.frestore(F,F_Check,i)
                for j= 1:$alg.period
                    $alg.fstore(F,F_Check_inner,t,j)
                    $(forloop.args[2])
                end
                for j= $alg.period:-1:1
                    t = $alg.frestore(F,F_Check_inner,j)
                    L_H .= L
                    F_H .= F
                    adjoint(F_H, L_H, F, L, t, h)
                    t = t - h
                end
            end
            F .= F_final
        end
    end
    esc(ex)
end

function optcontrol(scheme, steps)
    header()
    println( "\n STEPS    -> number of time steps to perform")
    println("SNAPS    -> number of checkpoints")
    println("INFO = 1 -> calculate only approximate solution")
    println("INFO = 2 -> calculate approximate solution + takeshots")
    println("INFO = 3 -> calculate approximate solution + all information ")
    println(" ENTER:   STEPS, SNAPS, INFO \n")


    h = 1.0/steps
    # F_final = Array{Float64, 1}(undef, 2)
    L = Array{Float64, 1}(undef, 2)
    L_H = Array{Float64, 1}(undef, 2)

    t = 0.0
    F_H = [1.0, 0.0]
    F = copy(F_H)

    # set_input(revolve, F)
    # set_output(revolve, F)

    @checkpoint scheme for i in 1:steps
        F_H[1] = F[1]
        F_H[2] = F[2]
        advance(F,F_H,t,h)
        t += h
    end


    # Enzyme crashes while differentiating this fully inlined function implementation
    # function tobedifferentiated(F, F_H)
    #     e = exp(1)
    #     for i in 1:steps
    #         F_H[1] = F[1]
    #         F_H[2] = F[2]
    #         # k0 = func(F_H,t)
    #             func_U = 2.0*((e^(3.0*t))-(e^3))/((e^(3.0*t/2.0))*(2.0+(e^3)))
    #             k0 = copy(F_H)
    #             k0[1] = 0.5*F_H[1]+ func_U
    #             k0[2] = F_H[1]*F_H[1]+0.5*(func_U*func_U)
    #         G = copy(F_H)
    #         G[1] = F_H[1] + h/2.0*k0[1]
    #         G[2] = F_H[2] + h/2.0*k0[2]
    #         # k1 = func(G,t+h/2.0)
    #             t_ = t+h/2.0
    #             func_U = 2.0*((e^(3.0*t_))-(e^3))/((e^(3.0*t_/2.0))*(2.0+(e^3)))
    #             k1 = copy(G)
    #             k1[1] = 0.5*G[1]+ func_U
    #             k1[2] = G[1]*G[1]+0.5*(func_U*func_U)
    #         F[1] = F_H[1] + h*k1[1]
    #         F[2] = F_H[2] + h*k1[2]
    #         t += h
    #     end
    # end
    # autodiff(tobedifferentiated, Duplicated(F, L), Duplicated(F_H, L_H))

    F_opt = Array{Float64, 1}(undef, 2)
    L_opt = Array{Float64, 1}(undef, 2)
    opt_sol(F_opt,1.0)
    opt_lambda(L_opt,0.0)
    println("\n\n")
    println("y_1*(1)  = " , F_opt[1] , " y_2*(1)  = " , F_opt[2])
    println("y_1 (1)  = " , F[1] , "  y_2 (1)  = " , F[2] , " \n\n")
    println("l_1*(0)  = " , L_opt[1] , "  l_2*(0)  = " , L_opt[2])
    println("l_1 (0)  = " , L[1]     , "  sl_2 (0)  = " , L[2] , " ")
    return F_opt, F, L_opt, L
end
