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

Any other scheme (a variable holding one, keyword arguments, `Online_r2`,
`EnzymeLLVM(...)`) is Checkpointing.jl's: the loop body becomes a closure that
`checkpoint_for` or `checkpoint_while` runs, and Checkpointing.jl, which must
be loaded, reverses it with that scheme.
"""
module CheckpointingCore

export @ad_checkpoint, checkpoint_for, checkpoint_while

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

const CHECKPOINTING = Base.PkgId(Base.UUID("eb46d486-4f9c-4c3d-b445-a617f2a2f1ca"), "Checkpointing")

function not_a_scheme(scheme)
    msg = if haskey(Base.loaded_modules, CHECKPOINTING)
        "@ad_checkpoint takes Revolve(k), Binomial(k), Periodic(k) or StoreAll() with a " *
        "literal budget, or one of Checkpointing.jl's schemes, not a $(typeof(scheme))"
    else
        "@ad_checkpoint with a $(typeof(scheme)) needs Checkpointing.jl: load it with " *
        "`using Checkpointing`. Without it, @ad_checkpoint takes Revolve(k), Binomial(k), " *
        "Periodic(k) or StoreAll() with a literal budget"
    end
    throw(ArgumentError(msg))
end

"""
    checkpoint_for(body, scheme, range)

Run `body(i)` for each `i` in `range`, reversed with `scheme`: what
`@ad_checkpoint` makes of a `for` loop whose scheme is not one of the
compiler's schedules. Checkpointing.jl adds the methods for its schemes.
"""
checkpoint_for(body, scheme, range) = not_a_scheme(scheme)

"""
    checkpoint_while(body, scheme)

Run `body()` until it returns `false`, reversed with `scheme`: what
`@ad_checkpoint` makes of a `while` loop whose scheme is not one of the
compiler's schedules. Checkpointing.jl adds the methods for its schemes.
"""
checkpoint_while(body, scheme) = not_a_scheme(scheme)

"""
    scheme_loop(scheme, loop::Expr)

`loop` as a closure run by `checkpoint_for` or `checkpoint_while` with
`scheme`, an expression evaluated where the loop is.
"""
function scheme_loop(scheme, loop::Expr)
    body = loop.args[2]
    if loop.head === :for
        iterator = loop.args[1].args[1]
        i = gensym("i")
        rng = gensym("range")
        return quote
            let
                # Bind the range once: interpolating it at each use would
                # evaluate the user's expression several times.
                $rng = $(loop.args[1].args[2])
                if !isa($rng, UnitRange{Int64})
                    error(
                        "@ad_checkpoint: only UnitRange{Int64} is supported, not $(typeof($rng))",
                    )
                end
                $(GlobalRef(@__MODULE__, :checkpoint_for))(
                    $i -> begin
                        $iterator = $i
                        $body
                    end,
                    $scheme,
                    $rng,
                )
            end
        end
    else
        return quote
            let
                $(GlobalRef(@__MODULE__, :checkpoint_while))(
                    () -> begin
                        $body
                        # The loop goes on while its condition holds.
                        return $(loop.args[1])
                    end,
                    $scheme,
                )
            end
        end
    end
end

"""
    @ad_checkpoint schedule loop

Mark `loop` (a `for` or `while` loop) for checkpointing. With
`Revolve(k)`, `Binomial(k)`, `Periodic(k)` or `StoreAll()` and a literal
budget `k`, the loop itself is unchanged and carries the loop annotation.
With any other scheme, which Checkpointing.jl provides and must be loaded
for, the loop body becomes a closure run by `checkpoint_for` or
`checkpoint_while`.
"""
macro ad_checkpoint(schedule, loop)
    loop isa Expr && loop.head in (:for, :while) ||
        throw(ArgumentError("@ad_checkpoint applies to a for or a while loop"))
    s = checkpoint_schedule(schedule)
    s === nothing || return esc(annotated_loop(s..., loop))
    return esc(scheme_loop(schedule, loop))
end

end
