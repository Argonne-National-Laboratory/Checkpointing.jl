# This is a Julia version of Solution of the optimal control problem 
# based on code written by Andrea Walther. See:
# Walther, Andrea, and Narayanan, Sri Hari Krishna. Extending the Binomial Checkpointing 
# Technique for Resilience. United States: N. p., 2016. https://www.osti.gov/biblio/1364654.


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

function timsteploop(F,F_H,t,h,steps)
    for i in 1:steps
        F_H[:] = F[:]
        advance(F,F_H,t,h)
        t += h
    end
    return
end

function main(info::Int, snaps::Int, steps::Int)
    header()
    println( "\n STEPS    -> number of time steps to perform")
    println("SNAPS    -> number of checkpoints")
    println("INFO = 1 -> calculate only approximate solution")
    println("INFO = 2 -> calculate approximate solution + takeshots")
    println("INFO = 3 -> calculate approximate solution + all information ")
    println(" ENTER:   STEPS, SNAPS, INFO \n")

    h = 1.0/steps
    t = 0.0
    F = Array{Float64, 1}(undef, 2)
    F_H = Array{Float64, 1}(undef, 2)
    F_final = Array{Float64, 1}(undef, 2)
    L = Array{Float64, 1}(undef, 2)
    L_H = Array{Float64, 1}(undef, 2)
    F[1] = 1.0
    F[2] = 0.0
    
    #TODO Differentiate this loop using Enzyme
    timsteploop(F,F_H,t,h,steps)

    F_final .= F
    F_opt = Array{Float64, 1}(undef, 2)
    L_opt = Array{Float64, 1}(undef, 2)
    opt_sol(F_opt,1.0)
    opt_lambda(L_opt,0.0)
    println("\n\n")
    println("y_1*(1)  = " , F_opt[1] , " y_2*(1)  = " , F_opt[2])
    println("y_1 (1)  = " , F_final[1] , "  y_2 (1)  = " , F_final[2] , " \n\n")
    println("l_1*(0)  = " , L_opt[1] , "  l_2*(0)  = " , L_opt[2])
    println("l_1 (0)  = " , L[1]     , "  sl_2 (0)  = " , L[2] , " ")
    return
end

main(3,3,10)