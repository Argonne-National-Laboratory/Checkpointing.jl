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

    mutable struct Counted
        x::Vector{Float64}
        n::Int
    end

    function checkpointed_while(s::Counted, alg, n::Int)
        s.n = 0
        @ad_checkpoint alg while s.n < n
            s.x .= s.x .+ 0.01 .* s.x .^ 2
            s.n += 1
        end
        return sum(s.x)
    end

    function plain_while_steps(s::Counted, n::Int)
        for _ = 1:n
            s.x .= s.x .+ 0.01 .* s.x .^ 2
        end
        return sum(s.x)
    end

    function while_gradient(f, args...)
        x = Counted([0.5, 0.8], 0)
        dx = Counted([0.0, 0.0], 0)
        _, primal =
            autodiff(Enzyme.ReverseWithPrimal, f, Active, Duplicated(x, dx), args...)
        return primal, dx.x, x.x
    end

    @testset "EnzymeLLVM(Online_r2($c)) over $n iterations" for c = 1:5,
        n in filter(<=(Checkpointing.online_r2_limit(c)), [1, 2, 3, 5, 10, 20, 30])

        want = while_gradient(plain_while_steps, Const(n))
        got = while_gradient(checkpointed_while, Const(EnzymeLLVM(Online_r2(c))), Const(n))
        @test got[1] ≈ want[1]
        @test got[2] ≈ want[2]
        @test got[3] == want[3]
    end

    # Issue #65: a while loop that reassigns a captured scalar.
    euler_rhs(t, y, B) = B * y
    function euler(y0, t_i, t_f, A, dt = 0.2)
        t = t_i + dt
        y = copy(y0)
        while t < t_f
            y .+= euler_rhs(t, y, A) .* dt
            t += dt
            t = min(t, t_f)
        end
        sum(y)
    end
    function euler_checkpointed(y0, t_i, t_f, A, alg, dt = 0.2)
        t = t_i + dt
        y = copy(y0)
        @ad_checkpoint alg while t < t_f
            y .+= euler_rhs(t, y, A) .* dt
            t += dt
            t = min(t, t_f)
        end
        sum(y)
    end

    @testset "Euler steps with a reassigned time (#65), Online_r2($c)" for c in (2, 3)
        y0 = [1.0 9.0; 1.0 9.0]
        A = [0 1.0; -100.0 0]
        dy0, dA = zero(y0), zero(A)
        autodiff(
            Reverse,
            euler,
            Active,
            Duplicated(y0, dy0),
            Const(0.0),
            Const(2.0),
            Duplicated(A, dA),
        )
        dy0_2, dA_2 = zero(y0), zero(A)
        autodiff(
            Reverse,
            euler_checkpointed,
            Active,
            Duplicated(y0, dy0_2),
            Const(0.0),
            Const(2.0),
            Duplicated(A, dA_2),
            Const(EnzymeLLVM(Online_r2(c))),
        )
        @test dy0_2 ≈ dy0
        @test dA_2 ≈ dA
    end

    # What a snapshot holds: Enzyme says what an iteration accesses.
    mutable struct Heat
        T::Vector{Float64}
        Tnext::Vector{Float64}
        κ::Vector{Float64}      # only read
        diag::Vector{Float64}   # not touched by the loop
        t::Float64
    end

    function heat_step!(h::Heat)
        T, Tn, κ = h.T, h.Tnext, h.κ
        @inbounds for k = 2:(length(T)-1)
            Tn[k] = T[k] + κ[k] * (T[k+1] - 2T[k] + T[k-1]) + 0.01 * sin(T[k] * h.t)
        end
        @inbounds for k = 2:(length(T)-1)
            T[k] = Tn[k]
        end
        h.t += 0.1
        return nothing
    end

    function heat(h::Heat, n, alg)
        @ad_checkpoint alg for i = 1:n
            heat_step!(h)
        end
        return sum(abs2, h.T) + sum(h.diag) * h.t
    end

    function heat_plain(h::Heat, n)
        for i = 1:n
            heat_step!(h)
        end
        return sum(abs2, h.T) + sum(h.diag) * h.t
    end

    function heat_gradient(f, args...)
        N = 100
        h = Heat([sin(k / 10) for k = 1:N], zeros(N), fill(0.2, N), ones(10N), 0.5)
        dh = Heat(zeros(N), zeros(N), zeros(N), zeros(10N), 0.0)
        autodiff(Reverse, f, Active, Duplicated(h, dh), args...)
        return dh, h
    end

    @testset "Snapshots of what an iteration accesses" begin
        want, want_h = heat_gradient(heat_plain, Const(30))
        bytes = Dict{Symbol,Int}()
        for snapshot in (:all, :accessed, :written)
            got, h = heat_gradient(
                heat,
                Const(30),
                Const(EnzymeLLVM(Revolve(4); snapshot = snapshot)),
            )
            @test got.T ≈ want.T
            @test got.κ ≈ want.κ
            @test got.diag ≈ want.diag
            @test got.t ≈ want.t
            @test h.T == want_h.T && h.t == want_h.t
            bytes[snapshot] = Checkpointing.LAST_SNAPSHOT_BYTES[]
        end
        # diag is not copied, and with :written neither is κ.
        @test bytes[:all] > bytes[:accessed] > bytes[:written]
    end

    # A step that accesses all of the body: masking would save nothing.
    mutable struct Rod
        T::Vector{Float64}
        Tnext::Vector{Float64}
    end

    function rod_step!(r::Rod)
        T, Tn = r.T, r.Tnext
        @inbounds for k = 2:(length(T)-1)
            Tn[k] = T[k] + 0.25 * (T[k+1] - 2T[k] + T[k-1])
        end
        @inbounds for k = 2:(length(T)-1)
            T[k] = Tn[k]
        end
        return nothing
    end

    function rod(r::Rod, n, alg)
        @ad_checkpoint alg for i = 1:n
            rod_step!(r)
        end
        return sum(abs2, r.T)
    end

    function rod_plain(r::Rod, n)
        for i = 1:n
            rod_step!(r)
        end
        return sum(abs2, r.T)
    end

    function rod_gradient(f, args...)
        N = 100
        r = Rod([sin(k / 10) for k = 1:N], zeros(N))
        dr = Rod(zeros(N), zeros(N))
        autodiff(Reverse, f, Active, Duplicated(r, dr), args...)
        return dr, r
    end

    @testset "Snapshots of the whole body when a mask saves nothing" begin
        want, want_r = rod_gradient(rod_plain, Const(30))
        bytes = Dict{Symbol,Int}()
        for snapshot in (:all, :accessed)
            got, r = rod_gradient(
                rod,
                Const(30),
                Const(EnzymeLLVM(Revolve(4); snapshot = snapshot)),
            )
            @test got.T ≈ want.T
            @test r.T == want_r.T
            bytes[snapshot] = Checkpointing.LAST_SNAPSHOT_BYTES[]
        end
        @test bytes[:accessed] == bytes[:all]
    end

    @test isempty(Checkpointing.LIVE_SCHEDULES)
end

end # module EnzymeLLVMTest
