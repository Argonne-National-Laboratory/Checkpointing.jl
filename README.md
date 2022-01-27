# Checkpointing

![CI](https://github.com/Argonne-National-Laboratory/Checkpointing.jl/workflows/Run%20tests/badge.svg?branch=main)

This package provides checkpointing schemes for adjoint computation using automatic differentiation (AD) of time stepping loops. The schemes are agnostic to the ADTool being used and can be easily interfaced with any Julia AD tool. Currently the package provides the following checkpointing schemes:

1. Revolve/Binomial checkpointing [1]
2. Periodic checkpointing

## Installation

```julia
add Checkpointing
```

## Interface with an AD Tool

Currently, `Checkpointing.jl` interfaces with an AD tool through the computation of a Jacobian by implementing a `jacobian` method. The following describes the interface for `ReverseDiff.jl`

```julia
using Checkpointing

struct ReverseDiffADTool <: AbstractADTool end

function Checkpointing.jacobian(tobedifferentiated, F_H, ::ReverseDiffADTool)
    return ReverseDiff.jacobian(tobedifferentiated, F_H)
end
```
The interfaces for `Diffractor.jl`, `Enzyme.jl`, `ForwardDiff.jl`, `ReverseDiff.jl`, and `Zygote.jl` are implemented in `examples/adtools.jl`.

## Usage

The macro `@checkpointing` covers the transformation of `for` loops with `1:steps` ranges where `steps` is the number of timesteps:

```julia
@checkpoint scheme adtool for i in 1:steps
    F_H = [F[1], F[2]]
    F = advance(F_H,t,h)
    t += h
end
```

where `adtool` is one of the interface AD tools (e.g. `ReverseDiffADTool()`) and scheme is a adjoint checkpointing scheme like for example revolve
```julia
function store(x::Vector, c::Vector, t::Int64, s::Int64)
    c[1,s] = x[1]
    c[2,s] = x[2]
    c[3,s] = t
    return
end

function restore(c, i)
    x = [c[1,i], c[2,i]]
    t = c[3,i]
    return x, t
end
Revolve(steps, checkpoints, store, store; verbose=verbose)
```
where `store` and `restore` are functions for storing and restoring `i`-th checkpoint `c[i]` with variables `x` and `t`. `steps` is the total number of timesteps while `checkpoints` is the number of available checkpoints.

## Future

The following features are planned for development:

* Integration with (`ChainRules.jl`)[https://github.com/JuliaDiff/ChainRules.jl] to generate rules for timestepping loops
* Online checkpointing schemes for adaptime timestepping
* Composition of checkpointing schemes
* Multi-level checkpointing schemes

[1] Andreas Griewank and Andrea Walther. 2000. Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation. ACM Trans. Math. Softw. 26, 1 (March 2000), 19–45. DOI:https://doi.org/10.1145/347837.347846

## Funding

This work is supported by the NSF Cyberinfrastructure for Sustained Scientific Innovation (CSSI) program project (DJ4Earth)[https://dj4earth.github.io/]

