# GPU tests. These live in their own environment so the CPU suite never has to
# install CUDA:
#
#     julia --project=test/gpu -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#     julia --project=test/gpu test/gpu/runtests.jl
using Test
using CUDA
using LinearAlgebra
using Checkpointing

if !CUDA.functional()
    @warn "CUDA is not functional on this machine; skipping the GPU tests."
else
    include(joinpath(@__DIR__, "..", "..", "examples", "heat_gpu.jl"))

    mutable struct DeviceState
        x::CuVector{Float64}
        y::CuVector{Float64}
        n::Int
    end

    @testset "GPU" begin
        @testset "checkpoint copies stay on the device and do not allocate" begin
            s = DeviceState(CUDA.rand(Float64, 1024), CUDA.rand(Float64, 1024), 3)
            slot = Checkpointing.checkpoint_alloc(s)
            @test slot.x isa CuVector{Float64}
            @test slot.x !== s.x
            s.x .= 1.0
            s.n = 7
            Checkpointing.checkpoint_copy!(slot, s)          # compile
            stats = CUDA.@timed Checkpointing.checkpoint_copy!(slot, s)
            @test stats.gpu_memstats.alloc_count == 0
            @test Array(slot.x) == Array(s.x)
            @test slot.n == 7
            # and the slot is a snapshot, not an alias
            s.x .= 2.0
            @test all(==(1.0), Array(slot.x))
        end

        # The pinned values are the CPU heat example's (test/runtests.jl): the
        # device formulation computes the same thing.
        @testset "heat on the device with $(nameof(typeof(scheme)))" for scheme in (
            Revolve(100),
            Periodic(100),
            Online_r2(100),
        )
            Tc, dTc = heat_device(scheme, 500)
            Tg, dTg = heat_device(scheme, 500; arraytype = CuArray)
            @test Tg ≈ Tc
            @test dTg ≈ dTc
            @test isapprox(norm(Tg), 66.21987468492061, atol = 1e-11)
            @test isapprox(norm(dTg), 6.970279349365908, atol = 1e-11)
        end
    end
end
