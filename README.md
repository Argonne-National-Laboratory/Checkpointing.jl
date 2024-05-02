# Checkpointing
[![][build-stable-img]][build-url] [![][docs-stable-img]][docs-stable-url] [![DOI](https://zenodo.org/badge/417181074.svg)](https://zenodo.org/badge/latestdoi/417181074)

This package provides checkpointing schemes for adjoint computations using automatic differentiation (AD) of time-stepping loops. Currently, we support the macro `@checkpoint_struct`, which differentiates and checkpoints a struct used in a while or for the loop with a `UnitRange`.

Each loop iteration is differentiated using [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl). We rely on external differentiation rule systems to integrate with AD tools applied to the code outside of the loop.

The schemes are agnostic to the AD tool being used and can be easily interfaced with any Julia AD tool. Currently, the package supports:

## Scheme
* Revolve/Binomial checkpointing [1]
* Periodic checkpointing
* Online r=2 checkpointing for while loops with a priori unknown number of iterations [2]

## Rules
* [ChainRulesCore.jl](https://juliadiff.org/ChainRulesCore.jl/stable/)
* [EnzymeRules.jl](https://enzyme.mit.edu/julia/stable/generated/custom_rule/)

## Storage
* ArrayStorage: Stores all checkpoints values in an array of type `Array`
* HDF5Storage: Stores all checkpoints values in an HDF5 file

## Installation

```julia
] add Checkpointing
```

## Related packages
* [TreeverseAlgorithm.jl](https://github.com/GiggleLiu/TreeverseAlgorithm.jl): Visualization of the Revolve algorithm
* [Burgers.jl](https://github.com/DJ4Earth/Burgers.jl): A showcase of checkpointing applied to an MPI parallelized 2D Burgers equation solver

## Usage: Example 1D heat equation

We present an example of a differentiated explicit 1D heat equation. Notice that the heat equation is a linear differential equation and does not require adjoint checkpointing. This example only illustrates the Checkpointing.jl API. The macro `@checkpointing_struct` covers the transformation of `for` loops with `UnitRange` ranges where `tsteps=500` is the number of time steps. As a checkpointing scheme, we use Revolve and use a maximum of only 4 snapshots. This implies that instead of requiring to save all 500 temperature fields for the gradient computation, we now only need 4. As a trade-off, recomputation is used to recompute intermediate temperature fields.

```julia
# Explicit 1D heat equation
using Checkpointing
using Enzyme
using Plots

mutable struct Heat
    Tnext::Vector{Float64}
    Tlast::Vector{Float64}
    n::Int
    λ::Float64
    tsteps::Int
end

function advance(heat)
    next = heat.Tnext
    last = heat.Tlast
    λ = heat.λ
    n = heat.n
    for i in 2:(n-1)
        next[i] = last[i] + λ*(last[i-1]-2*last[i]+last[i+1])
    end
    return nothing
end


function sumheat(heat::Heat, chkpscheme::Scheme, tsteps::Int64)
    # AD: Create shadow copy for derivatives
    @checkpoint_struct chkpscheme heat for i in 1:tsteps
        heat.Tlast .= heat.Tnext
        advance(heat)
    end
    return reduce(+, heat.Tnext)
end

function heat(scheme::Scheme, tsteps::Int)
    n = 100
    Δx=0.1
    Δt=0.001
    # Select μ such that λ ≤ 0.5 for stability with μ = (λ*Δt)/Δx^2
    λ = 0.5

    # Create object from struct. tsteps is not needed for a for-loop
    heat = Heat(zeros(n), zeros(n), n, λ, tsteps)
    # Shadow copy for Enzyme
    dheat = Heat(zeros(n), zeros(n), n, λ, tsteps)

    # Boundary conditions
    heat.Tnext[1]   = 20.0
    heat.Tnext[end] = 0

    # Compute gradient
    autodiff(Enzyme.ReverseWithPrimal, sumheat, Duplicated(heat, dheat), Const(scheme), Const(tsteps))

    return heat.Tnext, dheat.Tnext[2:end-1]
end
tsteps = 500
T, dT = heat(Revolve{Heat}(tsteps,4), tsteps)
# Plot function values
plot(T)
# Plot gradient with respect of sum(T[end]) with respect to T[1].
plot(dT)
```
## Future

The following features are planned for development:

* Support checkpoints on GPUs

[1] Andreas Griewank and Andrea Walther, Algorithm 799: Revolve: An Implementation of Checkpointing for the Reverse or Adjoint Mode of Computational Differentiation. ACM Trans. Math. Softw. 26, 1 (March 2000), 19–45. DOI: [10.1145/347837.347846](https://doi.org/10.1145/347837.347846)

[2] Philipp Stumm and Andrea Walther, New Algorithms for Optimal Online Checkpointing, 2010, DOI: [10.1137/080742439](https://doi.org/10.1137/080742439)

## Funding

This work is supported by the NSF Cyberinfrastructure for Sustained Scientific Innovation (CSSI) program project [DJ4Earth](https://dj4earth.github.io/)

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://Argonne-National-Laboratory.github.io/Checkpointing.jl/
[build-url]: https://github.com/Argonne-National-Laboratory/Checkpointing.jl/actions?query=workflow/actions?query=workflow
[build-stable-img]: https://github.com/Argonne-National-Laboratory/Checkpointing.jl/workflows/Run%20tests/badge.svg?branch=main
