using Test
using Checkpointing
using LinearAlgebra
using DataStructures
# All tested AD tools
using ForwardDiff, ReverseDiff, Zygote, Enzyme
# Include all the AD tool interfaces through `jacobian()`
include("../examples/adtools.jl")

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

    @testset "Testing Jacobian interface..." begin
        include("jacobian.jl")
    end
    @testset "Test optcontrol" begin
        @testset "AD Tool $adtool" for adtool in [EnzymeADTool(), ForwardDiffADTool(), ReverseDiffADTool(), ZygoteADTool()]
            @testset "Testing Revolve..." begin
                include("../examples/deprecated/optcontrol.jl")
                global steps = 100
                global snaps = 3
                global info = 0

                function store(F_H, F_C,t, i)
                    F_C[1,i] = F_H[1]
                    F_C[2,i] = F_H[2]
                    F_C[3,i] = t
                    return
                end

                function restore(F_C, i)
                    F_H = [F_C[1,i], F_C[2,i]]
                    t = F_C[3,i]
                    return F_H, t
                end
                revolve = Revolve{Nothing}(steps, snaps, store, restore; verbose=info)
                F_opt, F_final, L_opt, L = optcontrol(revolve, steps, adtool)
                @test isapprox(F_opt, F_final, rtol=1e-4)
                @test isapprox(L_opt, L, rtol=1e-4)
            end


            @testset "Testing Periodic..." begin
                include("../examples/deprecated/optcontrol.jl")
                global steps = 100
                global snaps = 4
                global info = 0

                function store(F_H, F_C,t, i)
                    F_C[1,i] = F_H[1]
                    F_C[2,i] = F_H[2]
                    F_C[3,i] = t
                    return
                end

                function restore(F_C, i)
                    F_H = [F_C[1,i], F_C[2,i]]
                    t = F_C[3,i]
                    return F_H, t
                end
                periodic = Periodic{Nothing}(steps, snaps, store, restore; verbose=info)
                F_opt, F_final, L_opt, L = optcontrol(periodic, steps, adtool)
                @test isapprox(F_opt, F_final, rtol=1e-4)
                @test isapprox(L_opt, L, rtol=1e-4)
            end

            @testset "Testing Online_r2..." begin
                include("../examples/optcontrolwhile.jl")
                # Enzyme segfaults if the garbage collector is enabled
                global steps = 100
                global snaps = 20
                global info = 0

                function store(F_H, F_C,t, i)
                    F_C[1,i] = F_H[1]
                    F_C[2,i] = F_H[2]
                    F_C[3,i] = t
                    return
                end

                function restore(F_C, i)
                    F_H = [F_C[1,i], F_C[2,i]]
                    t = F_C[3,i]
                    return F_H, t
                end
                online = Online_r2{Nothing}(snaps, store, restore)
                F_opt, F_final, L_opt, L = optcontrolwhile(online, steps, adtool)
                @test isapprox(F_opt, F_final, rtol=1e-4)
                @test isapprox(L_opt, L, rtol=1e-4)
            end
        end
    end

    @testset "Test optcontrol" begin
        include("../examples/optcontrol.jl")
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
    @testset "Test heat example" begin
        include("../examples/heat.jl")
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
    @testset "Test HDF5 storage using heat example" begin
        include("../examples/heat.jl")
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
    @testset "Test box model example" begin
        include("../examples/box_model.jl")

        @testset "Testing Revolve..." begin
            steps = 10000
            snaps = 100
            info = 0

            revolve = Revolve{Box}(steps, snaps; verbose=info)
            T, dT = box_for(revolve, steps)
            @test isapprox(T, 21.41890316892692)
            @test isapprox(dT[2][5], 0.00616139595759519)

        end

        @testset "Testing All Enzyme..." begin
            steps = 10000
            snaps = 100
            info = 0

            periodic = Periodic{Box}(steps, snaps; verbose=info)
            T, dT = box_for(periodic, steps)
            @test isapprox(T, 21.41890316892692)
            @test isapprox(dT[2][5], 0.00616139595759519)
        end

        @testset "Testing Online_r2..." begin
            steps = 10000
            snaps = 500
            info = 0
            online = Online_r2{Box}(snaps; verbose=info)
            T, dT = box_while(online, steps)
            @test isapprox(T, 21.41890316892692)
            @test isapprox(dT[2][5], 0.00616139595759519)
        end
    end
end
