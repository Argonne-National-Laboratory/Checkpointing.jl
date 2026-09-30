# @ad_checkpoint EnzymeLLVM(scheme): the loop is reversed by Enzyme's LLVM core
# under this package's schemes. Needs an Enzyme whose EnzymeCore has
# checkpoint_for and whose libEnzyme lowers it; skipped otherwise.
module EnzymeLLVMTest

using Checkpointing
using Enzyme
using Enzyme: EnzymeCore
using Base.Libc: Libdl
using Test

const SUPPORTED =
    isdefined(EnzymeCore, :checkpoint_for) &&
    Libdl.dlsym_e(Libdl.dlopen(Enzyme.API.libEnzyme), :EnzymeLowerCheckpointMarkers) !=
    C_NULL

if !SUPPORTED
    @info "Skipping the EnzymeLLVM tests: this Enzyme cannot lower EnzymeCore.checkpoint_for"
else
    mutable struct Traced
        x::Vector{Float64}
    end

    step!(s::Traced, i) = (s.x .= s.x .+ 0.01 .* i .* s.x .^ 2; nothing)

    function checkpointed_for(s::Traced, alg, r::UnitRange{Int})
        @ad_checkpoint alg for i in r
            step!(s, i)
        end
        return sum(s.x)
    end

    function plain_for(s::Traced, r::UnitRange{Int})
        for i in r
            step!(s, i)
        end
        return sum(s.x)
    end

    function gradient(f, args...)
        x = Traced([0.5, 0.8])
        dx = Traced([0.0, 0.0])
        _, primal =
            autodiff(Enzyme.ReverseWithPrimal, f, Active, Duplicated(x, dx), args...)
        return primal, dx.x, x.x
    end

    @testset "EnzymeLLVM($(nameof(typeof(scheme)))(3)) over $r" for scheme in (
            Revolve(3),
            Periodic(3),
        ),
        r in (1:10, 3:12, 1:9, 1:1, 1:0)

        want_primal, want_grad, want_final = gradient(plain_for, Const(r))
        primal, grad, final =
            gradient(checkpointed_for, Const(EnzymeLLVM(scheme)), Const(r))
        @test primal ≈ want_primal
        @test grad ≈ want_grad
        @test final == want_final
    end

    # The inner scheme in the model, as in multilevel.jl.
    mutable struct Nested{S}
        x::Vector{Float64}
        inner::S
    end

    function loops(s::Nested, outer, it1::Int, it2::Int)
        @ad_checkpoint outer for i = 1:it1
            @ad_checkpoint s.inner for j = 1:it2
                s.x .= s.x .+ 0.01 .* sin.(s.x .* (i + j))
            end
        end
        return sum(abs2, s.x)
    end

    # The inner scheme captured by the outer loop body.
    function loops_captured(s::Nested, outer, inner, it1::Int, it2::Int)
        @ad_checkpoint outer for i = 1:it1
            @ad_checkpoint inner for j = 1:it2
                s.x .= s.x .+ 0.01 .* sin.(s.x .* (i + j))
            end
        end
        return sum(abs2, s.x)
    end

    function loops_plain(s::Nested, it1::Int, it2::Int)
        for i = 1:it1, j = 1:it2
            s.x .= s.x .+ 0.01 .* sin.(s.x .* (i + j))
        end
        return sum(abs2, s.x)
    end

    function nested_gradient(mode, f, inner, args...)
        x = Nested([2.0, 3.0, 4.0], inner)
        dx = Nested([0.0, 0.0, 0.0], inner)
        _, primal = autodiff(mode, f, Active, Duplicated(x, dx), args...)
        return primal, dx.x, x.x
    end

    @testset "Multilevel: $name" for (name, outer, inner) in [
        ("EnzymeLLVM in EnzymeLLVM", EnzymeLLVM(Periodic(2)), EnzymeLLVM(Revolve(2))),
        ("EnzymeLLVM in EnzymeRules", Periodic(2), EnzymeLLVM(Revolve(2))),
        ("EnzymeRules in EnzymeLLVM", EnzymeLLVM(Periodic(2)), Revolve(2)),
    ]
        mode = Enzyme.ReverseWithPrimal
        want = nested_gradient(mode, loops_plain, inner, Const(4), Const(5))
        got = nested_gradient(mode, loops, inner, Const(outer), Const(4), Const(5))
        @test got[1] ≈ want[1]
        @test got[2] ≈ want[2]
        @test got[3] == want[3]
        # A body capturing both the state and a scheme mixes active and
        # constant data, which Enzyme needs runtime activity for.
        mode = Enzyme.set_runtime_activity(mode)
        got = nested_gradient(
            mode,
            loops_captured,
            nothing,
            Const(outer),
            Const(inner),
            Const(4),
            Const(5),
        )
        @test got[1] ≈ want[1]
        @test got[2] ≈ want[2]
        @test got[3] == want[3]
    end

    @test isempty(Checkpointing.LIVE_SCHEDULES)
end

end # module EnzymeLLVMTest
