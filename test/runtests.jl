using Test
using Checkpointing
using LinearAlgebra
using Enzyme

@testset "Checkpointing.jl" begin
    @testset "Enzyme..." begin
        @test Enzyme.EnzymeRules.has_rrule_from_sig(
            Base.signature_type(Checkpointing.checkpoint_for, Tuple{Any,Any,Any}),
        )
        @test Enzyme.EnzymeRules.has_rrule_from_sig(
            Base.signature_type(Checkpointing.checkpoint_while, Tuple{Any,Any}),
        )
        include("speelpenning.jl")
        errf, errg = main()
        @test isapprox(errf, 0.0; atol = 1e-15)
        @test isapprox(errg, 0.0; atol = 1e-15)
    end
    @testset "Revolve..." begin
        global steps = 50
        global checkpoints = 7
        global verbose = 0
        include("../examples/printaction.jl")
        revolve = main(steps, checkpoints)

        @test revolve.numfwd == 105
        @test revolve.numstore == 28
        @test revolve.numinv == 177
    end
    @testset "Testing optcontrol..." begin
        include("../examples/optcontrol.jl")
        @testset for scheme in [:Revolve, :Periodic]
            steps = 100
            snaps = 4
            info = 0

            F, L, F_opt, L_opt =
                muoptcontrol(eval(scheme)(snaps, verbose = info), steps, snaps)
            @test isapprox(F_opt, F, rtol = 1e-4)
            @test isapprox(L_opt, L, rtol = 1e-4)
        end
    end
    @testset "Testing heat example" begin
        include("../examples/heat.jl")
        @testset "$scheme" for scheme in [:Revolve, :Periodic, :Online_r2]
            steps = 500
            snaps = 100
            info = 0

            T, dT = heat(eval(scheme)(snaps; verbose = info), steps)

            @test isapprox(norm(T), 66.21987468492061, atol = 1e-11)
            @test isapprox(norm(dT), 6.970279349365908, atol = 1e-11)
        end
        @testset "Testing HDF5 storage using heat example" begin
            @testset "$scheme" for scheme in [:Revolve, :Periodic, :Online_r2]
                steps = 500
                snaps = 100
                info = 0

                T, dT =
                    heat(eval(scheme)(snaps; verbose = info, storage = :HDF5Storage), steps)

                @test isapprox(norm(T), 66.21987468492061, atol = 1e-11)
                @test isapprox(norm(dT), 6.970279349365908, atol = 1e-11)
            end
        end
    end
    @testset "Test box model example" begin
        include("../examples/box_model.jl")
        @testset "$scheme" for scheme in [:Revolve, :Periodic, :Online_r2]
            steps = 10000
            snaps = 500
            info = 0

            T, dT = box(eval(scheme)(snaps; verbose = info), steps)
            @test isapprox(T, 21.41890316892692)
            @test isapprox(dT[5], 0.00616139595759519)

        end
    end
    @testset "Multilevel" begin
        include("multilevel.jl")
    end
    @testset "Test writing checkpoints out" begin
        include("output_chkp.jl")
        @testset "$scheme" for scheme in [:Revolve, :Periodic]
            output_chkp(scheme)
        end
    end
end
