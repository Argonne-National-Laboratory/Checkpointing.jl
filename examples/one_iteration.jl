using OrdinaryDiffEq
using Plots

#Constants
const g = 9.81
L = 1.0

#Initial Conditions
u₀ = [0,π/2]
tspan = (0.0,6.3)

#Define the problem
function simplependulum(du,u,p,t)
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -(g/L)*sin(θ)
end

#Pass to solvers
dt = 0.01
prob = ODEProblem(simplependulum, u₀, tspan)
sol = solve(prob, Euler(); dt=dt)
plot(sol,linewidth=2,title ="Simple Pendulum Problem", xaxis = "Time", yaxis = "Height", label = ["\\theta" "d\\theta"])

prob_ts = ODEProblem(simplependulum, u₀, (0, dt))
integrator = init(prob_ts, Euler(); dt=dt,save_everystep=false)

steps = Int((tspan[2]-tspan[1])/dt)
u = zeros(steps, 2)
u[1,:] .= u₀[:]

macro one_iteration(args)
    # head is a `for` symbol
    # second argument is a loop iterator which now goes from 1 to 1
    # last argument is the body
    return Expr(args.head, :(i = 1:1), args.args[2])
end

@one_iteration for i in 1:steps-1
	set_u!(integrator, u[i,:])
    step!(integrator)
    u[i+1, :] = integrator.u
end

# Only one loop iteration was executed and is printed
plot(u)
