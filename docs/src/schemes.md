# Checkpointing Schemes
A checkpointing scheme may support either for or while loops or both. Each scheme has to define a struct with a constructor derived from the abstract type `Scheme`.
```julia
mutable struct MyScheme{FT} <: Scheme where {FT}
```
Schemes use a two-phase instantiation. The user creates a scheme parameterized by `Nothing`:
```julia
function MyScheme(checkpoints::Integer; kwargs...)
    return MyScheme{Nothing}(...)
end
```
During the AD reverse pass, `instantiate` is called to create the actual scheme parameterized by the loop body function type `FT`:
```julia
function instantiate(::Type{FT}, scheme::MyScheme{Nothing}, steps::Int) where {FT}
    return MyScheme{FT}(...)
end
```

A scheme has then to implement the reverse pass with the checkpointing scheme that will replace the for or while loop:
```julia
function rev_checkpoint_for(
    config,
    body_input::Function,
    dbody::Function,
    alg::MyScheme{FT},
    range,
) where {FT} end
function rev_checkpoint_while(
    config,
    body_input::Function,
    dbody::Function,
    alg::MyScheme{FT},
) where {FT}
```
The reverse pass for the checkpointing scheme will be called with the Enzyme config `config`, the loop body closure `body_input`, the shadow (derivative) closure `dbody`, the scheme `alg`, and the loop range `range` (for `for` loops).
