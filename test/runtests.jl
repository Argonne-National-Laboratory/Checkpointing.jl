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

@testset "Testing Revolve..." begin
    global steps = 100
    global snaps = 3
    global info = 3
    include("../examples/optcontrol.jl")
    F_opt, F_final, L_opt, L = main(steps, snaps, info)
    @test isapprox(F_opt, F_final, rtol=1e-4)
    @test isapprox(L_opt, L, rtol=1e-4)
end