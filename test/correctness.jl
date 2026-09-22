# Gradients through every scheme checked against Enzyme on the same loop without
# checkpointing. The bodies are nonlinear on purpose: the adjoint of each step
# then depends on the state it is taken at, so a reverse sweep that replays the
# wrong steps, the wrong indices or too few steps is caught -- not just one that
# hands the adjoint the wrong index.
using Checkpointing
using Enzyme
using Test

mutable struct Traced
    x::Vector{Float64}
    n::Int
end

step!(s::Traced, i) = (s.x .= s.x .+ 0.01 .* i .* s.x .^ 2; nothing)

function checkpointed_for(s::Traced, scheme::Scheme, r::UnitRange{Int})
    @ad_checkpoint scheme for i in r
        step!(s, i)
    end
    return sum(s.x)
end

function checkpointed_while(s::Traced, scheme::Scheme, n::Int)
    s.n = 0
    @ad_checkpoint scheme while s.n < n
        step!(s, 1)
        s.n += 1
    end
    return sum(s.x)
end

function plain_for(s::Traced, r::UnitRange{Int})
    for i in r
        step!(s, i)
    end
    return sum(s.x)
end

function plain_while_steps(s::Traced, n::Int)
    for _ = 1:n
        step!(s, 1)
    end
    return sum(s.x)
end

function gradient(f, args...)
    x = Traced([0.5, 0.8], 0)
    dx = Traced([0.0, 0.0], 0)
    _, primal = autodiff(Enzyme.ReverseWithPrimal, f, Active, Duplicated(x, dx), args...)
    return primal, dx.x, x.x
end

@testset "Gradients match an uncheckpointed loop" begin
    # 10 steps do not divide evenly into 3 checkpoints, 3:12 does not start at
    # 1, 1:9 divides evenly as a control, and 1:1 and 1:0 are the degenerate
    # cases.
    @testset "$(nameof(typeof(scheme)))(3) over $r" for scheme in (Revolve(3), Periodic(3)),
        r in (1:10, 3:12, 1:9, 1:1, 1:0)

        want_primal, want_grad, want_final = gradient(plain_for, Const(r))
        primal, grad, final = gradient(checkpointed_for, Const(scheme), Const(r))
        @test primal ≈ want_primal
        @test grad ≈ want_grad
        # The primal is driven by the scheme's schedule now, so check the loop
        # really leaves the state where a plain loop would.
        @test final == want_final
    end
    # Online_r2 only works in a certain regime: the offline phase assumes the
    # online phase filled every checkpoint, so very short loops are out of
    # scope. These lengths are inside it for c = 3.
    @testset "Online_r2(3) over $n iterations" for n in (8, 10, 30)
        want_primal, want_grad, want_final = gradient(plain_while_steps, Const(n))
        primal, grad, final = gradient(checkpointed_while, Const(Online_r2(3)), Const(n))
        @test primal ≈ want_primal
        @test grad ≈ want_grad
        @test final == want_final
    end
end
