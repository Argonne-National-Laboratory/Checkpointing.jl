"""
    CheckpointingCore

How a loop asks to be checkpointed, with no dependencies, so that model code
can annotate its time loops without depending on Enzyme or Checkpointing.jl,
and so that the differentiation tools that honor the annotation (Enzyme.jl,
Reactant) can read it without depending on each other.

    @ad_checkpoint Revolve(4) for t in 1:n
        step!(model)
    end

leaves the loop as it is, a plain loop, and marks it with a loop annotation:

    Expr(:loopinfo, (Symbol("enzyme.checkpoint"), :revolve, 4))

at the end of its body, which Julia attaches to the loop's back edge as LLVM
loop metadata. Running the loop is unaffected: no pass reads the annotation.
Differentiating it, Enzyme checkpoints the loop with the schedule named, one
iteration a step; Reactant's `@trace` lowers it to Enzyme-MLIR's
checkpointing attributes. The schedules and budgets are those of Enzyme's
`enzyme/checkpoint_schedule.h`:

  - `Revolve(k)`: Griewank and Walther's Revolve with `k` checkpoints
  - `Binomial(k)`: Enzyme-MLIR's binomial schedule with `k` checkpoints
  - `Periodic(k)`: `k` segments, whose starts are checkpointed
  - `StoreAll()`: a checkpoint before every iteration

The budget is the number of checkpoints that live through the reverse sweep,
never a segment length; without one, the schedule takes its default
(about √n). The schedule is read from the macro's argument as written, so the
budget must be a literal integer.
"""
module CheckpointingCore

export @ad_checkpoint

"""The name of the loop annotation, as Enzyme reads it."""
const LOOP_ANNOTATION = Symbol("enzyme.checkpoint")

const SCHEDULES = Dict{Symbol,Symbol}(
    :Revolve => :revolve,
    :Binomial => :binomial,
    :Periodic => :periodic,
    :StoreAll => :store_all,
)

"""
    checkpoint_schedule(ex) -> Union{Tuple{Symbol,Int},Nothing}

The schedule and budget a schedule expression as written names, such as
`Revolve(4)` or `Checkpointing.Periodic(10)`, or `nothing` if it names none
with a literal budget (a variable, keyword arguments, another scheme). A
budget of 0 asks for the schedule's default.
"""
function checkpoint_schedule(ex)
    ex isa Expr && ex.head === :call || return nothing
    f = ex.args[1]
    if f isa Expr && f.head === :. && f.args[end] isa QuoteNode
        f = f.args[end].value
    end
    f isa Symbol || return nothing
    name = get(SCHEDULES, f, nothing)
    name === nothing && return nothing
    args = ex.args[2:end]
    if isempty(args)
        return (name, 0)
    elseif length(args) == 1 && args[1] isa Integer && name !== :store_all
        return (name, Int(args[1]))
    end
    return nothing
end

"""
    loop_annotation(schedule::Symbol, budget::Integer)

The `Expr(:loopinfo, ...)` that marks a loop for checkpointing; it goes last
in the loop's body.
"""
loop_annotation(schedule::Symbol, budget::Integer) =
    Expr(:loopinfo, (LOOP_ANNOTATION, schedule, Int(budget)))

"""
    annotated_loop(schedule::Symbol, budget::Integer, loop::Expr)

`loop`, a `for` or `while` loop, marked for checkpointing.
"""
function annotated_loop(schedule::Symbol, budget::Integer, loop::Expr)
    loop.head in (:for, :while) ||
        throw(ArgumentError("@ad_checkpoint applies to a for or a while loop"))
    body = loop.args[2]
    return Expr(loop.head, loop.args[1], Expr(:block, body, loop_annotation(schedule, budget)))
end

"""
    loop_checkpoint(loop::Expr) -> Union{Tuple{Symbol,Int},Nothing}

The schedule and budget a loop (as `@ad_checkpoint` leaves it, its macros
expanded) is marked with, or `nothing`: for a frontend that lowers the
annotation itself, as Reactant's `@trace` does.
"""
function loop_checkpoint(loop::Expr)
    loop.head in (:for, :while) || return nothing
    body = loop.args[2]
    body isa Expr && body.head === :block || return nothing
    for ex in body.args
        if ex isa Expr && ex.head === :loopinfo
            for info in ex.args
                if info isa Tuple && length(info) == 3 && info[1] === LOOP_ANNOTATION
                    return (info[2]::Symbol, info[3]::Int)
                end
            end
        end
    end
    return nothing
end

"""
    @ad_checkpoint schedule loop

Mark `loop` (a `for` or `while` loop) for checkpointing with `schedule`, one
of `Revolve(k)`, `Binomial(k)`, `Periodic(k)` or `StoreAll()` with a literal
budget `k`. The loop itself is unchanged. Checkpointing.jl's `@ad_checkpoint`
takes any of its schemes, and expands to this for these.
"""
macro ad_checkpoint(schedule, loop)
    s = checkpoint_schedule(schedule)
    s === nothing && throw(
        ArgumentError(
            "CheckpointingCore.@ad_checkpoint takes Revolve(k), Binomial(k), Periodic(k) " *
            "or StoreAll() with a literal budget, not $(schedule); Checkpointing.jl's " *
            "@ad_checkpoint takes other schemes",
        ),
    )
    return esc(annotated_loop(s..., loop))
end

end
