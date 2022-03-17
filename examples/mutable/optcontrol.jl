# This is a Julia version of Solution of the optimal control problem
# based on code written by Andrea Walther. See:
# Walther, Andrea, and Narayanan, Sri Hari Krishna. Extending the Binomial Checkpointing
# Technique for Resilience. United States: N. p., 2016. https://www.osti.gov/biblio/1364654.

using Checkpointing

mutable struct Model
    F::Vector{Float64}
    F_H::Vector{Float64}
    t::Float64
    h::Float64
end

include("optcontrolfunc.jl")

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

function muoptcontrol(scheme, steps)
    println( "\n STEPS    -> number of time steps to perform")
    println("SNAPS    -> number of checkpoints")
    println("INFO = 1 -> calculate only approximate solution")
    println("INFO = 2 -> calculate approximate solution + takeshots")
    println("INFO = 3 -> calculate approximate solution + all information ")
    println(" ENTER:   STEPS, SNAPS, INFO \n")


    # F   : output
    # F_H : input
    # L   : seed the output adjoint
    # L   : set input adjoint to 0
    F = [1.0, 0.0]
    F_H = [0.0, 0.0]
    L = [0.0, 1.0]
    L_H = [0.0, 0.0]
    t = 0.0
    h = 1.0/steps
    model = Model(F, F_H, t, h)
    # t and h are not active so, set their adjoints to zero.
    shadowmodel = Model(L, L_H, 0.0, 0.0)

    @checkpoint_mutable scheme adtool model shadowmodel for i in 1:steps
        model.F_H .= model.F
        advance(model)
        model.t += h 
    end

    F = model.F
    L = shadowmodel.F

    F_opt = Array{Float64, 1}(undef, 2)
    L_opt = Array{Float64, 1}(undef, 2)
    opt_sol(F_opt,1.0)
    opt_lambda(L_opt,0.0)
    println("\n\n")
    println("y_1*(1)  = " , F_opt[1] , " y_2*(1)  = " , F_opt[2])
    println("y_1 (1)  = " , F[1] , "  y_2 (1)  = " , F[2] , " \n\n")
    println("l_1*(0)  = " , L_opt[1] , "  l_2*(0)  = " , L_opt[2])
    println("l_1 (0)  = " , L[1]     , "  sl_2 (0)  = " , L[2] , " ")
    return model, shadowmodel, F_opt, L_opt
end

# main(10,3,3)