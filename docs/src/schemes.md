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
function rev_checkpoint_for(config, tape, dbody::Function, alg::MyScheme, range) end
function rev_checkpoint_while(config, tape, dbody::Function, alg::MyScheme) end
```
The reverse half is called with the Enzyme config `config`, the `tape` returned by the forward half, the shadow closure `dbody`, the scheme `alg`, and the loop range `range` (for `for` loops).

These four functions are unexported, so a new scheme extends `Checkpointing.fwd_checkpoint_for` and its siblings. A `for` loop body is called as `body(i)` with the current element of `range`. A `while` loop body is called as `body()` and returns the loop condition. Checkpoints are written with `Checkpointing.save!(alg.storage, body, i)` and read back with `Checkpointing.load!(body, alg.storage, i)`; see [Storage](storage.md). A single iteration `i` of a `for` loop is differentiated as below; for a `while` loop, drop the `Const(i)` argument.
```julia
Enzyme.autodiff(
    EnzymeCore.set_runtime_activity(Reverse, config),
    Duplicated(body, dbody),
    Const,
    Const(i),
)
```
