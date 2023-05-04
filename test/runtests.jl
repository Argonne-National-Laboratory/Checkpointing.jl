using Test
using Checkpointing
using LinearAlgebra
using DataStructures
# All tested AD tools
using Zygote, Enzyme
adtools = ["zygote", "enzyme"]

@testset "Testing Checkpointing.jl" begin
    @testset "Testing Enzyme..." begin
        include("speelpenning.jl")
        errf, errg = main()
        @test isapprox(errf, 0.0; atol = 1e-15)
        @test isapprox(errg, 0.0; atol = 1e-15)
    end
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

    @testset "Test optcontrol with $adtool" for adtool in adtools
        include("../examples/optcontrol_$(adtool).jl")
        @testset "Testing Revolve..." begin
            steps = 100
            snaps = 3
            info = 0

            revolve = Revolve{Model}(steps, snaps; verbose=info)
            F, L, F_opt, L_opt = muoptcontrol(revolve, steps)
            @test isapprox(F_opt, F, rtol=1e-4)
            @test isapprox(L_opt, L, rtol=1e-4)
        end

        @testset "Testing Periodic..." begin
            steps = 100
            snaps = 4
            info = 0

            periodic = Periodic{Model}(steps, snaps; verbose=info)
            F, L, F_opt, L_opt = muoptcontrol(periodic, steps)
            @test isapprox(F_opt, F, rtol=1e-4)
            @test isapprox(L_opt, L, rtol=1e-4)
        end
    end
    @testset "Test heat example with $adtool" for adtool in adtools
        include("../examples/heat_$(adtool).jl")
        @testset "Testing Revolve..." begin
            steps = 500
            snaps = 4
            info = 0

            revolve = Revolve{Heat}(steps, snaps; verbose=info)
            T, dT = heat_for(revolve, steps)

            @test isapprox(norm(T), 66.21987468492061, atol=1e-11)
            @test isapprox(norm(dT), 6.970279349365908, atol=1e-11)
        end

        @testset "Testing Periodic..." begin
            steps = 500
            snaps = 4
            info = 0

            periodic = Periodic{Heat}(steps, snaps; verbose=info)
            T, dT = heat_for(periodic, steps)

            @test isapprox(norm(T), 66.21987468492061, atol=1e-11)
            @test isapprox(norm(dT), 6.970279349365908, atol=1e-11)
        end

        @testset "Testing Online_r2..." begin
            steps = 500
            snaps = 100
            info = 0
            online = Online_r2{Heat}(snaps; verbose=info)
            T, dT = heat_while(online, steps)

            @test isapprox(norm(T), 66.21987468492061, atol=1e-11)
            @test isapprox(norm(dT), 6.970279349365908, atol=1e-11)
        end
    end
    @testset "Test HDF5 storage using heat example with $adtool" for adtool in adtools
        include("../examples/heat_$adtool.jl")
        @testset "Testing Revolve..." begin
            steps = 500
            snaps = 4
            info = 0

            revolve = Revolve{Heat}(steps, snaps; storage=HDF5Storage{Heat}(snaps), verbose=info)
            T, dT = heat_for(revolve, steps)

            @test isapprox(norm(T), 66.21987468492061, atol=1e-11)
            @test isapprox(norm(dT), 6.970279349365908, atol=1e-11)
        end

        @testset "Testing Periodic..." begin
            steps = 500
            snaps = 4
            info = 0

            periodic = Periodic{Heat}(steps, snaps; storage=HDF5Storage{Heat}(snaps), verbose=info)
            T, dT = heat_for(periodic, steps)

            @test isapprox(norm(T), 66.21987468492061, atol=1e-11)
            @test isapprox(norm(dT), 6.970279349365908, atol=1e-11)
        end

        @testset "Testing Online_r2..." begin
            steps = 500
            snaps = 100
            info = 0
            online = Online_r2{Heat}(snaps; storage=HDF5Storage{Heat}(snaps), verbose=info)
            T, dT = heat_while(online, steps)

            @test isapprox(norm(T), 66.21987468492061, atol=1e-11)
            @test isapprox(norm(dT), 6.970279349365908, atol=1e-11)
        end
    end
    @testset "Test box model example with $(adtool)" for adtool in adtools
        include("../examples/box_model_$(adtool).jl")

        @testset "Testing Revolve..." begin
            steps = 10000
            snaps = 100
            info = 0

            revolve = Revolve{Box}(steps, snaps; verbose=info)
            T, dT = box_for(revolve, steps)
            @test isapprox(T, 21.41890316892692)
            @test isapprox(dT[5], 0.00616139595759519)

        end

        @testset "Testing Periodic..." begin
            steps = 10000
            snaps = 100
            info = 0

            periodic = Periodic{Box}(steps, snaps; verbose=info)
            T, dT = box_for(periodic, steps)
            @test isapprox(T, 21.41890316892692)
            @test isapprox(dT[5], 0.00616139595759519)
        end

        @testset "Testing Online_r2..." begin
            steps = 10000
            snaps = 500
            info = 0
            online = Online_r2{Box}(snaps; verbose=info)
            T, dT = box_while(online, steps)
            @test isapprox(T, 21.41890316892692)
            @test isapprox(dT[5], 0.00616139595759519)
        end
    end
end
