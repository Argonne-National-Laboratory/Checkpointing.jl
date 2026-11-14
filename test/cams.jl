# CAMS: the schedule on its own, then gradients through it. Reuses `Traced`,
# `step!`, the loops and the gradient helpers from correctness.jl and batch.jl.
using Checkpointing
using Enzyme
using Test
import Checkpointing:
    CAMSTable,
    cams_actions,
    cams_advance,
    cams_store,
    cams_restore,
    cams_tape,
    cams_adjoint,
    cams_adjoint_tape

# Run a schedule on step numbers alone. Checks that every step is adjoined
# once, in order, from a state or a tape that is actually there, and returns
# the number of recomputed steps, the table's prediction of it, and the peak
# memory in units.
function replay(steps, units, stage_units; exact = nothing)
    table = CAMSTable(steps, units, stage_units; exact = exact)
    position = 0
    states = Dict{Int,Int}()
    tapes = Dict{Int,Int}()
    front = steps
    forward = 0
    peak = 0
    for act in cams_actions(table)
        if act.kind == cams_advance
            @test act.a == position + 1
            forward += act.b - act.a + 1
            position = act.b
        elseif act.kind == cams_store
            states[act.a] = position
        elseif act.kind == cams_restore
            position = states[act.a]
        elseif act.kind == cams_tape
            @test act.a == position + 1
            @test !haskey(tapes, act.b)
            tapes[act.b] = act.a
            forward += 1
            position = act.a
        elseif act.kind == cams_adjoint
            @test act.a == position + 1 == front
            forward += 1
            position = act.a
            front -= 1
        else
            @test tapes[act.b] == act.a == front
            delete!(tapes, act.b)
            front -= 1
        end
        # A state is needed until every step behind it has been adjoined.
        filter!(p -> p.second < front, states)
        peak = max(peak, length(states) + stage_units * length(tapes))
    end
    @test front == 0
    @test isempty(tapes)
    return forward - steps, table.R[table.units+1, steps+1], peak
end

@testset "CAMS" begin
    @testset "Recomputation counts" begin
        # One unit can only hold the initial state.
        @test cams_recomputations(10, 1) == 45
        # Zhang & Constantinescu, Figs. 6 and 7.
        @test cams_recomputations(10, 6; stage_units = 2) == 8
        @test cams_recomputations(300, 30; stage_units = 2) == 358
        # A tape too large to use leaves Revolve's count, t m - C(s + t, t - 1).
        @test cams_recomputations(10, 3; stage_units = 100) == 15
        @test cams_recomputations(300, 30; stage_units = 1000) == 568
        # Enough memory to tape every step but the last.
        @test cams_recomputations(50, 49 * 3; stage_units = 3) == 0
        @test cams_recomputations(50, 49 * 3 - 1; stage_units = 3) > 0
        @test cams_recomputations(1, 1) == 0
        @test cams_recomputations(0, 1) == 0
    end
    @testset "Schedules: $units units, $stage_units per tape" for units in
                                                                  (1, 2, 3, 5, 8, 13, 40),
        stage_units = 1:4

        for steps in (1, 2, 3, 7, 20, 61)
            cost, predicted, peak = replay(steps, units, stage_units)
            @test cost == predicted
            @test peak <= units
            # More memory or a smaller tape never costs more.
            @test cost <= cams_recomputations(steps, max(units - 1, 1); stage_units)
            @test cost <= cams_recomputations(steps, units; stage_units = stage_units + 1)
            # The narrowed search used for long loops still gives a schedule
            # that is valid and within budget.
            near, predicted, peak = replay(steps, units, stage_units; exact = false)
            @test near == predicted
            @test peak <= units
            @test near >= cost
        end
    end
    @testset "Gradients: CAMS($units; stage_units = $stage_units) over $r" for units in (
            1,
            2,
            4,
            7,
            30,
        ),
        stage_units in (1, 2, 3),
        r in (1:10, 3:12, 1:1, 1:0)

        scheme = CAMS(units; stage_units = stage_units)
        want_primal, want_grad, want_final = gradient(plain_for, Const(r))
        primal, grad, final = gradient(checkpointed_for, Const(scheme), Const(r))
        @test primal ≈ want_primal
        @test grad ≈ want_grad
        # The reverse sweep recomputes on the loop body itself and has to put
        # the final state back.
        @test final == want_final
    end
    @testset "Batched: CAMS($units), N = $N" for units in (2, 6), N in (2, 3)
        r = 1:10
        scheme = CAMS(units; stage_units = 1)
        check_lanes(
            checkpointed_for!,
            plain_for!,
            N,
            (Const(scheme), Const(r)),
            (Const(r),),
        )
    end
    @testset "Nested in and around Revolve" begin
        for (outer, inner) in
            ((CAMS(3), Revolve(2)), (Revolve(2), CAMS(3)), (CAMS(2), CAMS(4)))
            x = Chkp([2.0, 3.0, 4.0], inner)
            dx = Chkp([0.0, 0.0, 0.0], inner)
            autodiff(
                Reverse,
                loops,
                Active,
                Duplicated(x, dx),
                Const(outer),
                Const(2),
                Const(5),
            )
            @test all(dx.x .== 1024.0)
        end
    end
    @testset "Arguments" begin
        @test_throws ArgumentError CAMS(0)
        @test_throws ArgumentError CAMS(3; stage_units = 0)
        @test_throws ArgumentError gradient(checkpointed_while, Const(CAMS(3)), Const(5))
    end
end
