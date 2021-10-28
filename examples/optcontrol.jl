# This is a Julia version of Solution of the optimal control problem
# based on code written by Andrea Walther. See:
# Walther, Andrea, and Narayanan, Sri Hari Krishna. Extending the Binomial Checkpointing
# Technique for Resilience. United States: N. p., 2016. https://www.osti.gov/biblio/1364654.

using Checkpointing

function func_U(t)
    e = 2.7182818
    return 2.0*((e^(3.0*t))-(e^3))/((e^(3.0*t/2.0))*(2.0+(e^3)))
end

function func(X,t)
    F = Array{Float64, 1}(undef, 2)
    F[1] = 0.5*X[1]+ func_U(t)
    F[2] = X[1]*X[1]+0.5*(func_U(t)*func_U(t))
    return F
end

function func_adj(BF, X)
    BX = Array{Float64, 1}(undef, 2)
    BX[1] = 0.5*BF[1]+2.0*X[1]*BF[2]
    BX[2] = 0.0
    return BX
end

function opt_sol(Y,t)
    e = 2.7182818
    Y[1] = (2.0*e^(3.0*t)+e^3)/(e^(3.0*t/2.0)*(2.0+e^3))
    Y[2] = (2.0*e^(3.0*t)-e^(6.0-3.0*t)-2.0+e^6)/((2.0+e^3)^2)
    return
end

function opt_lambda(L,t)
    e = 2.7182818
    L[1] = (2.0*e^(3-t)-2.0*e^(2.0*t))/(e^(t/2.0)*(2+e^3))
    L[2] = 1.0
    return
end

function advance(F,F_H,t,h)
    k0 = func(F_H,t)
    G = Array{Float64, 1}(undef, 2)
    G[1] = F_H[1] + h/2.0*k0[1]
    G[2] = F_H[2] + h/2.0*k0[2]
    k1 = func(G,t+h/2.0)
    F[1] = F_H[1] + h*k1[1]
    F[2] = F_H[2] + h*k1[2]
    return
end

function store(F_H, F_C,t, i)
    F_C[1,i] = F_H[1]
    F_C[2,i] = F_H[2]
    F_C[3,i] = t
    return
end

function restore(F_H, F_C, i)
    F_H[1] = F_C[1,i]
    F_H[2] = F_C[2,i]
    t = F_C[3,i]
    return t
end

function adjoint(L_H,F_H,L,t,h)
    k0 = func(F_H,t)
    G = Array{Float64, 1}(undef, 2)
    Bk1 = Array{Float64, 1}(undef, 2)
    Bk0 = Array{Float64, 1}(undef, 2)
    G[1] = F_H[1] + h/2.0*k0[1]
    G[2] = F_H[2] + h/2.0*k0[2]
    k1 = func(G,t+h/2.0)
    L[1] = L_H[1]
    L[2] = L_H[2]
    Bk1[1] = h*L_H[1]
    Bk1[2] = h*L_H[2]
    BG = func_adj(Bk1,G)
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

        println("\n \n Using Binomial Offline Checkpointing for the approximate solution: \n")
        return
end


macro checkpoint(alg, forloop)
    # esc(args)
    ex = quote
        storemap = Dict{Int32,Int32}()
        check = 0
        F_Check = Array{Float64, 2}(undef, 3, snaps)
        while true
            next_action = next_action!($alg)
            if (next_action.actionflag == Checkpointing.store)
                check = check+1
                storemap[next_action.iteration-1]=check
                store(F,F_Check,t,check)
            elseif (next_action.actionflag == Checkpointing.forward)
                for j= next_action.startiteration:(next_action.iteration - 1)
                    $(forloop.args[2])
                end
            elseif (next_action.actionflag == Checkpointing.firstuturn)
                F_H[1] = F[1]
                F_H[2] = F[2]
                advance(F_final,F_H,t,h)
                L[1] = 0.0
                L[2] = 1.0
                t = 1.0-h
                L_H[1] = L[1]
                L_H[2] = L[2]
                adjoint(L_H,F_H,L,t,h)
            elseif (next_action.actionflag == Checkpointing.uturn)
                L_H[1] = L[1]
                L_H[2] = L[2]
                adjoint(L_H,F,L,t,h)
                t = t - h
                if haskey(storemap,next_action.iteration-1-1)
                    delete!(storemap,next_action.iteration-1-1)
                    check=check-1
                end
            elseif (next_action.actionflag == Checkpointing.restore)
                t = restore(F,F_Check,storemap[next_action.iteration-1])
            elseif next_action.actionflag == Checkpointing.done
                break
            end
        end
        # $args
    end
    esc(ex)
end

function main(steps, snaps, info)
    header()
    println( "\n STEPS    -> number of time steps to perform")
    println("SNAPS    -> number of checkpoints")
    println("INFO = 1 -> calculate only approximate solution")
    println("INFO = 2 -> calculate approximate solution + takeshots")
    println("INFO = 3 -> calculate approximate solution + all information ")
    println(" ENTER:   STEPS, SNAPS, INFO \n")

    revolve = Revolve(steps, snaps; verbose=info)
    h = 1.0/steps
    F_final = Array{Float64, 1}(undef, 2)
    L = Array{Float64, 1}(undef, 2)
    L_H = Array{Float64, 1}(undef, 2)

    t = 0.0
    F = [1.0, 0.0]
    F_H = similar(F)

    @checkpoint revolve for i in 1:steps
        F_H[1] = F[1]
        F_H[2] = F[2]
        advance(F,F_H,t,h)
        t += h
    end

    F_opt = Array{Float64, 1}(undef, 2)
    L_opt = Array{Float64, 1}(undef, 2)
    opt_sol(F_opt,1.0)
    opt_lambda(L_opt,0.0)
    println("\n\n")
    println("y_1*(1)  = " , F_opt[1] , " y_2*(1)  = " , F_opt[2])
    println("y_1 (1)  = " , F_final[1] , "  y_2 (1)  = " , F_final[2] , " \n\n")
    println("l_1*(0)  = " , L_opt[1] , "  l_2*(0)  = " , L_opt[2])
    println("l_1 (0)  = " , L[1]     , "  sl_2 (0)  = " , L[2] , " ")
    return F_opt, F_final, L_opt, L
end
