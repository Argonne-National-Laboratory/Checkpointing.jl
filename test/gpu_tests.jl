using CUDA
using LinearAlgebra
using Adapt
using GPUArraysCore: allowscalar

# Enable scalar indexing on GPU arrays for testing.
# Examples like optcontrol and box_model use scalar indexing internally.
allowscalar(true)

@testset "GPU Support" begin
    @testset "ArrayStorage with CuArrays" begin
        mutable struct GPUState
            x::CuVector{Float64}
            y::CuVector{Float64}
        end
        state = GPUState(CUDA.ones(10), CUDA.zeros(10))
        body = (i) -> begin
            state.y .= state.x .* 2.0
        end

        # ArrayStorage should work with closures containing GPU arrays
        storage = ArrayStorage{typeof(body)}(3)
        cp = deepcopy(body)
        save!(storage, cp, 1)
        loaded = load(body, storage, 1)
        @test Array(loaded.state.x) == ones(10)
    end

    @testset "HDF5Storage rejects GPU arrays" begin
        mutable struct GPUStateHDF5
            x::CuVector{Float64}
        end
        state = GPUStateHDF5(CUDA.ones(5))
        body = (i) -> begin
            state.x .= state.x .+ 1.0
        end

        @test_throws ArgumentError Checkpointing.check_no_gpu_arrays(body)
    end

    @testset "Adapt.jl integration" begin
        include("../examples/heat.jl")

        h = Heat(zeros(10), zeros(10), 10, 0.5, 100)
        h_gpu = adapt(CuArray, h)
        @test h_gpu.Tnext isa CuVector{Float64}
        @test h_gpu.Tlast isa CuVector{Float64}
        @test h_gpu.n == 10
        @test h_gpu.λ == 0.5
    end

    @testset "GPU heat example" begin
        include("../examples/heat_gpu.jl")
        @testset "$scheme" for scheme in [:Revolve, :Periodic, :Online_r2]
            steps = 500
            snaps = 100
            T, dT = heat_gpu(eval(scheme)(snaps); arraytype = CuArray)
            @test isapprox(norm(T), 66.21987468492061, atol = 1.0e-11)
            @test isapprox(norm(dT), 6.970279349365908, atol = 1.0e-11)
        end
    end

    @testset "GPU optcontrol example" begin
        include("../examples/optcontrol.jl")

        function muoptcontrol_gpu(scheme, steps, snaps)
            F = CuVector([1.0, 0.0])
            F_H = CuVector([0.0, 0.0])
            t = 0.0
            h = 1.0 / steps
            model = Model(F, F_H, t, h)
            bmodel = Model(CUDA.zeros(2), CUDA.zeros(2), 0.0, 0.0)

            function foo(model::Model)
                @ad_checkpoint scheme for i in 1:steps
                    model.F_H .= model.F
                    advance(model)
                    model.t += model.h
                end
                return model.F[2]
            end
            autodiff(Enzyme.Reverse, Const(foo), Duplicated(model, bmodel))

            F_opt = Array{Float64, 1}(undef, 2)
            L_opt = Array{Float64, 1}(undef, 2)
            opt_sol(F_opt, 1.0)
            opt_lambda(L_opt, 0.0)
            return Array(model.F), Array(bmodel.F), F_opt, L_opt
        end

        @testset "$scheme" for scheme in [:Revolve, :Periodic]
            steps = 100
            snaps = 4
            F, L, F_opt, L_opt = muoptcontrol_gpu(eval(scheme)(snaps), steps, snaps)
            @test isapprox(F_opt, F, rtol = 1.0e-4)
            @test isapprox(L_opt, L, rtol = 1.0e-4)
        end
    end

    @testset "GPU box model example" begin
        include("../examples/box_model.jl")

        function box_gpu(scheme::Scheme, tsteps::Int)
            Tbar = [20.0; 1.0; 1.0]
            Sbar = [35.5; 34.5; 34.5]

            box_model = adapt(CuArray, Box(copy([Tbar; Sbar]), copy([Tbar; Sbar]), zeros(6), zeros(6), 0))
            dbox = adapt(CuArray, Box(zeros(6), zeros(6), zeros(6), zeros(6), 0))

            autodiff(
                Enzyme.ReverseWithPrimal,
                Const(timestepper),
                Duplicated(box_model, dbox),
                Const(scheme),
                Const(tsteps),
            )
            return box_model.out_now[1], Array(dbox.in_old)
        end

        @testset "$scheme" for scheme in [:Revolve, :Periodic, :Online_r2]
            steps = 10000
            snaps = 500
            T, dT = box_gpu(eval(scheme)(snaps), steps)
            @test isapprox(T, 21.41890316892692)
            @test isapprox(dT[5], 0.00616139595759519)
        end
    end
end
