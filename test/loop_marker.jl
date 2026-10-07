# @ad_checkpoint on a for loop with a literal Revolve(k) or Periodic(k), when
# the loaded Enzyme.jl checkpoints marked loops: the loop stays a plain loop
# marked with CheckpointingCore's annotation, and Enzyme reverses it. The
# gradient must be that of the loop without @ad_checkpoint, and of the same
# loop through the closure path.
using Checkpointing, Enzyme, Test
import CheckpointingCore

mutable struct MarkerState
    u::Vector{Float64}
    t::Float64
end

function marker_step!(s::MarkerState)
    s.u .= s.u .+ 0.1 .* sin.(s.u .* s.t)
    s.t += 0.1
    return nothing
end

function marker_revolve(s, n)
    @ad_checkpoint Revolve(3) for i in 1:n
        marker_step!(s)
    end
    return sum(abs2, s.u) * s.t
end

function marker_periodic(s, n)
    @ad_checkpoint Periodic(3) for i in 1:n
        marker_step!(s)
    end
    return sum(abs2, s.u) * s.t
end

function marker_plain(s, n)
    for i in 1:n
        marker_step!(s)
    end
    return sum(abs2, s.u) * s.t
end

function marker_closure(s, n)
    scheme = Revolve(3)   # not a literal: the closure path
    @ad_checkpoint scheme for i in 1:n
        marker_step!(s)
    end
    return sum(abs2, s.u) * s.t
end

function marker_grad(f, n)
    s = MarkerState(collect(range(0.1, 1, 8)), 0.3)
    ds = MarkerState(zeros(8), 0.0)
    autodiff(Reverse, f, Active, Duplicated(s, ds), Const(n))
    return (ds.u, ds.t, s.u, s.t)
end

@testset "@ad_checkpoint on a loop marker" begin
    @test Checkpointing.enzyme_marks_loops()
    ex = macroexpand(
        Main,
        Meta.parse("@ad_checkpoint Revolve(3) for i in 1:n; marker_step!(s); end"),
    )
    @test CheckpointingCore.loop_checkpoint(ex) == (:revolve, 3)
    ex = macroexpand(Main, Meta.parse("@ad_checkpoint scheme for i in 1:n; marker_step!(s); end"))
    @test CheckpointingCore.loop_checkpoint(ex) === nothing

    for n in (1, 7, 20)
        want = marker_grad(marker_plain, n)
        for f in (marker_revolve, marker_periodic, marker_closure)
            got = marker_grad(f, n)
            @test got[1] ≈ want[1] rtol = 1e-12
            @test got[2] ≈ want[2] rtol = 1e-12
            @test got[3] == want[3] && got[4] == want[4]
        end
    end
end
