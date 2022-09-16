# Checkpointing
[![][build-stable-img]][build-url] [![][docs-stable-img]][docs-stable-url] [![DOI](https://zenodo.org/badge/417181074.svg)](https://zenodo.org/badge/latestdoi/417181074)

This package provides checkpointing schemes for adjoint computations using automatic differentiation (AD) of time stepping loops. Currently, we support the macro `@checkpoint_struct`, which differentiates and checkpoints a struct used in the loop. Each loop iteration is differentiated using [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl). We rely on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) to integrate with AD tools applied to the code outside of the loop.

The schemes are agnostic to the AD tool being used and can be easily interfaced with any Julia AD tool. Currently, the package provides the following checkpointing schemes:

1. Revolve/Binomial checkpointing [1]
2. Periodic checkpointing
3. Online r=2 checkpointing for while loops with a priori unknown number of iterations [2]

## Installation

```julia
add Checkpointing
```

## Usage: Example 1D heat equation

We present an example code where Zygote is used to differentiate the implementation of the explicit 1D heat equation. The macro `@checkpointing_struct` covers the transformation of `for` loops with `1:tsteps` ranges where `tsteps=500` is the number of timesteps. As a checkpointing scheme, we use Revolve and use a maximum of only 4 snapshots. This implies that instead of requiring to save all 500 temperature fields for the gradient computation, we now only need 4. As a trade-off, recomputation is used to recompute intermediate temperature fields.

```julia
# Explicit 1D heat equation
using Checkpointing
using Plots
using Zygote

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


function sumheat(heat::Heat, chkpt::Scheme)
    @checkpoint_struct revolve heat for i in 1:tsteps
        heat.Tlast .= heat.Tnext
        advance(heat)
    end
    return reduce(+, heat.Tnext)
end

n = 100
Δx=0.1
Δt=0.001
# Select μ such that λ ≤ 0.5 for stability with μ = (λ*Δt)/Δx^2
λ = 0.5
# time steps
tsteps = 500

# Create object from struct
heat = Heat(zeros(n), zeros(n), n, λ, tsteps)

# Boundary conditions
heat.Tnext[1]   = 20.0
heat.Tnext[end] = 0

# Set up AD
# Number of available snapshots
snaps = 4
verbose = 0
revolve = Revolve{Heat}(tsteps, snaps; verbose=verbose)

# Compute gradient
g = Zygote.gradient(sumheat, heat, revolve)

# Plot function values
plot(heat.Tnext)
# Plot gradient with respect to sum(T).
plot(g[1].Tnext[2:end-1])
```
## Future

The following features are planned for development:

* Online checkpointing schemes for adaptive timestepping
* Composition of checkpointing schemes
* Multi-level checkpointing schemes

[1] Andreas Griewank and Andrea Walther, Algorithm 799: Revolve: An Implementation of Checkpointing for the Reverse or Adjoint Mode of Computational Differentiation. ACM Trans. Math. Softw. 26, 1 (March 2000), 19–45. DOI: [10.1145/347837.347846](https://doi.org/10.1145/347837.347846)

[2] Philipp Stumm and Andrea Walther, New Algorithms for Optimal Online Checkpointing, 2010, DOI: [10.1137/080742439](https://doi.org/10.1137/080742439)

## Funding

This work is supported by the NSF Cyberinfrastructure for Sustained Scientific Innovation (CSSI) program project [DJ4Earth](https://dj4earth.github.io/)

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://Argonne-National-Laboratory.github.io/Checkpointing.jl/
[build-url]: https://github.com/Argonne-National-Laboratory/Checkpointing.jl/actions?query=workflow/actions?query=workflow
[build-stable-img]: https://github.com/Argonne-National-Laboratory/Checkpointing.jl/workflows/Run%20tests/badge.svg?branch=main
