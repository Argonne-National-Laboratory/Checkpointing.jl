using Test
using Checkpointing
using LinearAlgebra

@testset "Testing Revolve..." begin
    global steps = 50
    global checkpoints = 7
    global verbose = 0
    include("../examples/printaction.jl")
    revolve = main(steps, checkpoints)

    @test revolve.numfwd == 105
    @test revolve.numstore == 28
    @test revolve.numinv == 177
end

include("../examples/optcontrol.jl")

@testset "Testing Revolve..." begin
    global steps = 100
    global snaps = 3
    global info = 3

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
    revolve = Revolve(steps, snaps, store, restore; verbose=info)
    F_opt, F_final, L_opt, L = optcontrol(revolve, steps)
    @test isapprox(F_opt, F_final, rtol=1e-4)
    @test isapprox(L_opt, L, rtol=1e-4)
end


 @testset "Testing Periodic..." begin
     global steps = 100
     global snaps = 4
     global info = 3

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
     periodic = Periodic(steps, snaps, store, restore; verbose=info)
     F_opt, F_final, L_opt, L = optcontrol(periodic, steps)
     @test isapprox(F_opt, F_final, rtol=1e-4)
     @test isapprox(L_opt, L, rtol=1e-4)
 end
 