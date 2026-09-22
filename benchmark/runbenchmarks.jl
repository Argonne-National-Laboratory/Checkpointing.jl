# CPU benchmark harness for Checkpointing.jl.
#
# Reports, per scheme, the wall time and allocation volume of one full
# forward+reverse sweep, plus the first-call (compile) time. Run with:
#
#   julia --project=benchmark benchmark/runbenchmarks.jl
#
# With preallocated checkpoint slots the reverse sweep allocates O(acp) rather
# than O(numfwd) copies of the state, so the allocation column should stay flat
# as the number of steps grows.

using Checkpointing
using Enzyme
using LinearAlgebra
using Printf

include("../examples/heat.jl")
include("../examples/box_model.jl")

"Run `f` once to force compilation, then `samples` more times; return the best."
function best_of(f, samples)
    f()
    best_t = Inf
    best_bytes = typemax(Int)
    best_allocs = typemax(Int)
    for _ = 1:samples
        stats = @timed f()
        best_t = min(best_t, stats.time)
        best_bytes = min(best_bytes, stats.bytes)
        # `@timed` reports gcstats only on some versions; count via Base
        best_allocs = min(best_allocs, stats.bytes)
    end
    return (time = best_t, bytes = best_bytes)
end

function bench_heat(scheme_ctor, steps, snaps; samples = 3)
    compile = @timed heat(scheme_ctor(snaps), steps)
    run = best_of(() -> heat(scheme_ctor(snaps), steps), samples)
    return (compile = compile.time, run = run.time, bytes = run.bytes)
end

function bench_box(scheme_ctor, steps, snaps; samples = 3)
    compile = @timed box(scheme_ctor(snaps), steps)
    run = best_of(() -> box(scheme_ctor(snaps), steps), samples)
    return (compile = compile.time, run = run.time, bytes = run.bytes)
end

function report(title, rows)
    println()
    println(title)
    println("-"^72)
    @printf("%-12s %12s %12s %14s\n", "scheme", "compile (s)", "run (s)", "alloc")
    println("-"^72)
    for (name, r) in rows
        @printf(
            "%-12s %12.3f %12.4f %14s\n",
            name,
            r.compile,
            r.run,
            Base.format_bytes(r.bytes)
        )
    end
end

function main()
    @info "Checkpointing.jl benchmarks" Threads.nthreads() VERSION

    heat_rows = Pair{String,Any}[]
    for (name, ctor) in ("Revolve" => Revolve, "Periodic" => Periodic)
        push!(heat_rows, name => bench_heat(ctor, 500, 100))
    end
    report("heat: 500 steps, 100 snapshots", heat_rows)

    box_rows = Pair{String,Any}[]
    for (name, ctor) in ("Revolve" => Revolve, "Periodic" => Periodic)
        push!(box_rows, name => bench_box(ctor, 10000, 500))
    end
    report("box_model: 10000 steps, 500 snapshots", box_rows)

    # Checkpoint-count sensitivity: how much forward work does the scheme do?
    # The primal drives the schedule up to the first u-turn, so the forward
    # sweep is not repeated: forward evaluations are `forwardcount + 1` (the
    # extra one finishes the primal past the snapshot taken for the first
    # u-turn), plus one evaluation inside each step's adjoint.
    println()
    println("Revolve forward evaluations (heat, 500 steps)")
    println("-"^72)
    @printf("%8s %14s %12s\n", "snaps", "forward evals", "overhead")
    println("-"^72)
    for snaps in (4, 10, 25, 50, 100)
        r = Checkpointing.Revolve{Nothing}(500, snaps)
        fwd = round(Int, Checkpointing.forwardcount(r)) + 1
        @printf("%8d %14d %11.2fx\n", snaps, fwd, fwd / 500)
    end
    return nothing
end

main()
