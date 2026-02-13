# Checkpointing
[![CI](https://github.com/Argonne-National-Laboratory/Checkpointing.jl/actions/workflows/action.yml/badge.svg?branch=main)](https://github.com/Argonne-National-Laboratory/Checkpointing.jl/actions/workflows/action.yml)
[![][docs-stable-img]][docs-stable-url]
[![DOI](https://zenodo.org/badge/417181074.svg)](https://zenodo.org/badge/latestdoi/417181074)

This package provides checkpointing schemes for adjoint computations using automatic differentiation (AD) of time-stepping loops. Currently, we support the macro `@ad_checkpoint`, which differentiates and checkpoints a mutable struct used in a while or for loop with a `UnitRange`.

Each loop iteration is differentiated using [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl). We rely on external differentiation rule systems to integrate with AD tools applied to the code outside of the loop.

The schemes are agnostic to the AD tool being used and can be easily interfaced with any Julia AD tool. Currently, the package supports:

## Scheme
* Revolve/Binomial checkpointing [1]
* Periodic checkpointing
* Online r=2 checkpointing for while loops with a priori unknown number of iterations [2]

## Rules
* [EnzymeRules.jl](https://enzyme.mit.edu/julia/stable/generated/custom_rule/)

## Storage
* ArrayStorage: Stores all checkpoints values in an array of type `Array` (supports GPU arrays)
* HDF5Storage: Stores all checkpoints values in an HDF5 file (CPU only)

## Installation

```julia
] add Checkpointing
```

## Related packages
* [TreeverseAlgorithm.jl](https://github.com/GiggleLiu/TreeverseAlgorithm.jl): Visualization of the Revolve algorithm
* [Burgers.jl](https://github.com/DJ4Earth/Burgers.jl): A showcase of checkpointing applied to an MPI parallelized 2D Burgers equation solver

## Usage: Example 1D heat equation

We present an example of a differentiated explicit 1D heat equation. Notice that the heat equation is a linear differential equation and does not require adjoint checkpointing. This example only illustrates the Checkpointing.jl API. The macro `@ad_checkpoint` covers the transformation of `for` loops with `UnitRange` ranges where `tsteps=500` is the number of time steps. As a checkpointing scheme, we use Revolve and use a maximum of only 4 snapshots. This implies that instead of requiring to save all 500 temperature fields for the gradient computation, we now only need 4. As a trade-off, recomputation is used to recompute intermediate temperature fields.

```julia
# Explicit 1D heat equation
using Checkpointing
using Enzyme
using Adapt

mutable struct Heat{T}
    Tnext::T
    Tlast::T
    n::Int
    λ::Float64
    tsteps::Int
end

function Adapt.adapt_structure(to, heat::Heat)
    Heat(adapt(to, heat.Tnext), adapt(to, heat.Tlast), heat.n, heat.λ, heat.tsteps)
end

function advance(heat::Heat)
    heat.Tnext[2:end-1] .= heat.Tlast[2:end-1] .+ heat.λ .* (
        heat.Tlast[1:end-2] .- 2 .* heat.Tlast[2:end-1] .+ heat.Tlast[3:end]
    )
    return nothing
end

function sumheat(heat::Heat, scheme::Union{Revolve,Periodic}, tsteps::Int64)
    @ad_checkpoint scheme for i = 1:tsteps
        heat.Tlast .= heat.Tnext
        advance(heat)
    end
    return reduce(+, heat.Tnext)
end

function heat(scheme::Scheme, tsteps::Int)
    n = 100
    λ = 0.5

    heat = Heat(zeros(n), zeros(n), n, λ, tsteps)
    dheat = Heat(zeros(n), zeros(n), n, λ, tsteps)

    heat.Tnext[1] = 20.0
    heat.Tnext[end] = 0

    autodiff(
        Enzyme.ReverseWithPrimal,
        sumheat,
        Duplicated(heat, dheat),
        Const(scheme),
        Const(tsteps),
    )

    return heat.Tnext, dheat.Tnext[2:(end-1)]
end
tsteps = 500
T, dT = heat(Revolve(4), tsteps)
```

## GPU Support

Checkpointing.jl supports GPU arrays from [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), and [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl). GPU array detection is provided via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), and structs can be moved to the GPU using [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl).

### How it works

The checkpointing loop runs on the **CPU host** — it is not a GPU kernel. Each iteration may internally launch GPU kernels, but the checkpoint orchestration (store, restore, forward, uturn) is host-side logic. The core mechanism is `deepcopy` on closures containing GPU arrays, which dispatches to the GPU backend's copy implementation (e.g., `CUDA.jl` defines `Base.deepcopy_internal` for `CuArray`).

### Making a struct GPU-compatible

Parameterize the array type and define `Adapt.adapt_structure`:

```julia
using Adapt

mutable struct MyModel{T}
    state::T
    n::Int
end

function Adapt.adapt_structure(to, m::MyModel)
    MyModel(adapt(to, m.state), m.n)
end
```

Then move to GPU with `adapt`:

```julia
using CUDA
model_cpu = MyModel(zeros(100), 100)
model_gpu = adapt(CuArray, model_cpu)  # state is now a CuVector on the GPU
```

### Writing GPU-compatible loop bodies

Use broadcasting instead of scalar loops so the computation runs on the GPU:

```julia
# CPU-only: scalar loop
for i = 2:(n-1)
    next[i] = last[i] + λ * (last[i-1] - 2*last[i] + last[i+1])
end

# GPU-compatible: broadcasting
next[2:end-1] .= last[2:end-1] .+ λ .* (
    last[1:end-2] .- 2 .* last[2:end-1] .+ last[3:end]
)
```

### Running on GPU

```julia
using CUDA
using Adapt

# Create on CPU, adapt to GPU
heat_model = adapt(CuArray, Heat(zeros(n), zeros(n), n, λ, tsteps))
dheat = adapt(CuArray, Heat(zeros(n), zeros(n), n, λ, tsteps))

autodiff(
    Enzyme.ReverseWithPrimal,
    sumheat,
    Duplicated(heat_model, dheat),
    Const(scheme),
    Const(tsteps),
)
```

### Storage compatibility

* **ArrayStorage** (default): Works with GPU arrays. Checkpoints stay on the device.
* **HDF5Storage**: Does **not** support GPU arrays (serialization cannot handle device pointers). An `ArgumentError` is thrown if GPU arrays are detected.

### GPU memory considerations

Each checkpoint stores a `deepcopy` of the full closure including all GPU arrays. With `Revolve(c)` using `c` checkpoints, this means `c` copies of all GPU state reside on the device. For large models, monitor GPU memory usage accordingly.

[1] Andreas Griewank and Andrea Walther, Algorithm 799: Revolve: An Implementation of Checkpointing for the Reverse or Adjoint Mode of Computational Differentiation. ACM Trans. Math. Softw. 26, 1 (March 2000), 19–45. DOI: [10.1145/347837.347846](https://doi.org/10.1145/347837.347846)

[2] Philipp Stumm and Andrea Walther, New Algorithms for Optimal Online Checkpointing, 2010, DOI: [10.1137/080742439](https://doi.org/10.1137/080742439)

## Funding

This work is supported by the NSF Cyberinfrastructure for Sustained Scientific Innovation (CSSI) program project [DJ4Earth](https://dj4earth.github.io/)

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://Argonne-National-Laboratory.github.io/Checkpointing.jl/
