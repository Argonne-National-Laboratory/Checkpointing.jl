# Vector mode: a `BatchDuplicated` loop body goes through one schedule, one set
# of checkpoints and one recomputation for all lanes. Each lane must match the
# unbatched checkpointed gradient and Enzyme on the plain loop. Reuses `Traced`,
# `step!` and the loops from correctness.jl.
using Checkpointing
using Enzyme
using Test

# A vector-valued output written into a duplicated argument, so every lane can
# be seeded differently.
function checkpointed_for!(out, s::Traced, scheme::Scheme, r::UnitRange{Int})
    @ad_checkpoint scheme for i in r
        step!(s, i)
    end
    out .= s.x .^ 2
    return nothing
end

function checkpointed_while!(out, s::Traced, scheme::Scheme, n::Int)
    s.n = 0
    @ad_checkpoint scheme while s.n < n
        step!(s, 1)
        s.n += 1
    end
    out .= s.x .^ 2
    return nothing
end

function plain_for!(out, s::Traced, r::UnitRange{Int})
    for i in r
        step!(s, i)
    end
    out .= s.x .^ 2
    return nothing
end

function plain_while_steps!(out, s::Traced, n::Int)
    for _ = 1:n
        step!(s, 1)
    end
    out .= s.x .^ 2
    return nothing
end

const SEEDS = ([1.0, 0.0], [0.0, 1.0], [0.5, -2.0])

# Returns the output, the final state and one gradient per seed.
function seeded_gradient(f!, seeds, args...)
    x = Traced([0.5, 0.8], 0)
    out = zeros(2)
    if length(seeds) == 1
        dx = Traced([0.0, 0.0], 0)
        autodiff(
            Reverse,
            f!,
            Const,
            Duplicated(out, copy(seeds[1])),
            Duplicated(x, dx),
            args...,
        )
        return out, x.x, (dx.x,)
    end
    dxs = ntuple(_ -> Traced([0.0, 0.0], 0), length(seeds))
    douts = map(copy, seeds)
    autodiff(
        Reverse,
        f!,
        Const,
        BatchDuplicated(out, douts),
        BatchDuplicated(x, dxs),
        args...,
    )
    return out, x.x, map(dx -> dx.x, dxs)
end

function batched_gradient(f, N, args...)
    x = Traced([0.5, 0.8], 0)
    dxs = ntuple(_ -> Traced([0.0, 0.0], 0), N)
    autodiff(Reverse, f, Active, BatchDuplicated(x, dxs), args...)
    return x.x, map(dx -> dx.x, dxs)
end

function check_lanes(f!, plain!, N, args, plain_args)
    seeds = SEEDS[1:N]
    want_out, want_final, _ = seeded_gradient(plain!, (SEEDS[1],), plain_args...)
    out, final, grads = seeded_gradient(f!, seeds, args...)
    @test out ≈ want_out
    @test final == want_final
    for k = 1:N
        _, _, (want_grad,) = seeded_gradient(plain!, (seeds[k],), plain_args...)
        _, _, (unbatched_grad,) = seeded_gradient(f!, (seeds[k],), args...)
        @test grads[k] ≈ want_grad
        @test grads[k] ≈ unbatched_grad
    end
    # The lanes are seeded differently, so they must not coincide.
    @test !(grads[1] ≈ grads[2])
end

@testset "Batched gradients match unbatched ones" begin
    @testset "$(nameof(typeof(scheme)))(3) over $r, N = $N" for scheme in
                                                                (Revolve(3), Periodic(3)),
        r in (1:10, 3:12),
        N in (2, 3)

        check_lanes(
            checkpointed_for!,
            plain_for!,
            N,
            (Const(scheme), Const(r)),
            (Const(r),),
        )
        # An active scalar return seeds every lane with one.
        _, want_grad, want_final = gradient(plain_for, Const(r))
        final, grads = batched_gradient(checkpointed_for, N, Const(scheme), Const(r))
        @test final == want_final
        @test all(g -> g ≈ want_grad, grads)
    end
    @testset "Online_r2($c) over $n iterations, N = $N" for c in (2, 3),
        n in (5, 8),
        N in (2, 3)

        check_lanes(
            checkpointed_while!,
            plain_while_steps!,
            N,
            (Const(Online_r2(c)), Const(n)),
            (Const(n),),
        )
        _, want_grad, want_final = gradient(plain_while_steps, Const(n))
        final, grads =
            batched_gradient(checkpointed_while, N, Const(Online_r2(c)), Const(n))
        @test final == want_final
        @test all(g -> g ≈ want_grad, grads)
    end
end
