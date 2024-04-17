using Checkpointing
using Enzyme
using ReverseDiff
using Test


function f(x)
    return [(x[1] * (x[2] - 2.0))^2, (x[1] - 1.0)^2 * x[2]^2]
end

function f2(x, res)
    y = f(x)
    copyto!(res, y)
    return nothing
end

x = [2.0, 6.0]

J = ReverseDiff.jacobian(f, x)

dx = copy(x)
dx = [0.0, 0.0]
y = [0.0, 0.0]
dy = [0.0, 1.0]
autodiff(f2, Duplicated(x, dx), Duplicated(y, dy))

J2 = similar(J)
fill!(J2, 0)
for i = 1:2
    x = [2.0, 6.0]
    fill!(dx, 0)
    fill!(y, 0)
    dy[i] = 1.0
    autodiff(f2, Duplicated(x, dx), Duplicated(y, dy))
    J2[i, :] = dx[:]
end

# Is correct, but gives the memmove warning
@test J2 ≈ J


include("../examples/optcontrol.jl")

@testset "Testing Enzyme in Revolve..." begin
    global steps = 100
    global snaps = 3
    global info = 1

    function store(F_H, F_C, t, i)
        F_C[1, i] = F_H[1]
        F_C[2, i] = F_H[2]
        F_C[3, i] = t
        return
    end

    function restore(F_C, i)
        F_H = [F_C[1, i], F_C[2, i]]
        t = F_C[3, i]
        return F_H, t
    end
    revolve = Revolve(steps, snaps, store, restore; verbose = info)
    F_opt, F_final, L_opt, L = optcontrol(revolve, steps, ReverseDiffADTool())
    # Check whether ReverseDiff works
    @test isapprox(F_opt, F_final, rtol = 1e-4)
    @test isapprox(L_opt, L, rtol = 1e-4)
    revolve = Revolve(steps, snaps, store, restore; verbose = info)
    F_opt_enzyme, F_final_enzyme, L_opt_enzyme, L_enzyme =
        optcontrol(revolve, steps, EnzymeADTool())
    @test isapprox(F_final, F_final_enzyme)
    # Returns wrong adjoints
    @test_broken isapprox(L, L_enzyme, rtol = 1e-4)
end
