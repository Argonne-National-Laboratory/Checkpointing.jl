# Checkpointing Schemes
A checkpointing scheme may support either for or while loops or both. Each scheme is a struct derived from the abstract type `Scheme` and parameterized by the type of the loop body closure.
```julia
mutable struct MyScheme{FT} <: Scheme end
```
Schemes use a two-phase instantiation. The user creates a scheme parameterized by `Nothing`, because the loop body does not exist yet:
```julia
function MyScheme(checkpoints::Integer; kwargs...)
    return MyScheme{Nothing}(...)
end
```
When differentiation reaches the loop, `instantiate` creates the actual scheme parameterized by the loop body closure type `FT`. For `for` loops it also receives the number of iterations. For `while` loops that number is only known once the loop has run, so it is not passed:
```julia
function instantiate(::Type{FT}, scheme::MyScheme{Nothing}, steps::Int) where {FT}
    return MyScheme{FT}(...)
end
function instantiate(::Type{FT}, scheme::MyScheme{Nothing}) where {FT}
    return MyScheme{FT}(...)
end
```

A scheme then implements the loop in two halves. The forward half runs in the augmented forward pass, where it replaces the loop itself. It runs the loop on `body`, so that `body` ends in the loop's final state, stores checkpoints wherever the schedule asks for them, and returns a tape for the reverse half:
```julia
function fwd_checkpoint_for(body::Function, alg::MyScheme, range) end
function fwd_checkpoint_while(body::Function, alg::MyScheme) end
```
The reverse half runs in the reverse pass. It restores checkpoints, recomputes iterations, and differentiates them one at a time, accumulating the adjoint in the shadow closure `dbody`:
```julia
function rev_checkpoint_for(config, tape, dbody, alg::MyScheme, range) end
function rev_checkpoint_while(config, tape, dbody, alg::MyScheme) end
```
The reverse half is called with the Enzyme config `config`, the `tape` returned by the forward half, the shadow closure `dbody`, the scheme `alg`, and the loop range `range` (for `for` loops). In Enzyme's vector mode (`BatchDuplicated`), `dbody` is a tuple with one shadow closure per lane; the schedule and the checkpoints only involve the primal `body`, so they serve all lanes at once.

These four functions are unexported, so a new scheme extends `Checkpointing.fwd_checkpoint_for` and its siblings. A `for` loop body is called as `body(i)` with the current element of `range`. A `while` loop body is called as `body()` and returns the loop condition. Checkpoints are written with `Checkpointing.save!(alg.storage, body, i)` and read back with `Checkpointing.load!(body, alg.storage, i)`; see [Storage](storage.md). A single iteration `i` of a `for` loop is differentiated as below, for a single shadow or for all lanes at once; for a `while` loop, drop the `Const(i)` argument.
```julia
Checkpointing.adjoint_step!(config, body, dbody, Const(i))
```

## Tapes as checkpoints: CAMS

[`Revolve`](@ref) stores the state an iteration starts from, so `adjoint_step!` runs that iteration forward once more before it differentiates it. [`CAMS`](@ref) can also keep the tape of an iteration, which is what Enzyme's reverse pass reads, and differentiate the iteration from that without running it again:
```julia
scheme = CAMS(20; stage_units = 1)
```
The budget is counted in units of one state checkpoint, and `stage_units` is the size of one tape in those units. The schedule decides per checkpoint whether a state or a tape is the better use of memory. With enough units to tape every iteration nothing is recomputed; with a tape too large to fit, the schedule is Revolve's.

Whether this is faster depends on what an iteration costs to run compared to what it costs to differentiate. An explicit step differentiated by Enzyme is cheap to run again and its tape is as large as the state or larger, so Revolve is the better choice. An implicit step whose nonlinear solve has a reverse rule from the implicit function theorem is the opposite case: running it is a Newton solve, its adjoint is one linear solve, and its tape is little more than the solution. On an implicit Euler discretization of Burgers' equation with 200 steps, where each step is one Newton-Krylov solve, the number of solves in the forward and reverse pass together is:

| units | `Revolve(units)` | `CAMS(units; stage_units = 2)` | `CAMS(units; stage_units = 1)` |
|---:|---:|---:|---:|
| 3 | 1506 | 1504 | 1487 |
| 5 | 991 | 974 | 914 |
| 10 | 723 | 652 | 561 |
| 20 | 579 | 488 | 394 |
| 40 | 559 | 385 | 364 |

[`cams_recomputations`](@ref) gives these counts, less the 200 solves of the loop itself, without running anything.

A scheme that differentiates in its forward half, as CAMS does to keep tapes from the first sweep, needs the Enzyme config and the shadow there. It extends the method of `fwd_checkpoint_for` that the rule calls, which takes the config and the annotated loop body and by default forwards to the three-argument method above:
```julia
function Checkpointing.fwd_checkpoint_for(config, body::Enzyme.Annotation, alg::MyScheme, range)
    dbody = Checkpointing.shadow(body)
    forward, reverse, Tape = Checkpointing.split_thunks(config, body.val, dbody, eltype(range))
    ...
end
```
`Checkpointing.tape_step!(forward, body, dbody, i)` runs iteration `i` and returns its tape, and `Checkpointing.adjoint_tape!(reverse, body, dbody, i, tape)` later accumulates the adjoint of that iteration from it. A tape holds everything the reverse pass reads, so the state of `body` may change in between, and it can be used once.
