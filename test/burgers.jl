# The README's Burgers example: the checkpointed sensitivity must match Enzyme on
# the same loop without checkpointing, past the time the shock forms.
using Checkpointing
using Enzyme
using Test

include("../examples/burgers.jl")

function burgers_plain(b::Burgers, steps::Int, k::Int)
    for i = 1:steps
        step!(b)
    end
    return b.u[k]
end

@testset "Burgers example" begin
    n = 128
    grid = ((1:n) .- 0.5) ./ n
    u0 = exp.(-((grid .- 0.3) ./ 0.1) .^ 2)
    steps = 200
    @testset "x* = $xp" for xp in (0.45, 0.6)
        k = argmin(abs.(grid .- xp))
        b = Burgers(u0)
        db = Enzyme.make_zero(b)
        autodiff(Reverse, burgers_plain, Duplicated(b, db), Const(steps), Const(k))
        @testset "$(nameof(typeof(scheme)))" for scheme in (Revolve(10), Periodic(10))
            @test sensitivity(u0, scheme, steps, k) ≈ db.u
        end
    end
end
