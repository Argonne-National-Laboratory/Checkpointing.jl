

# Explicit 1D heat equation

```@example heat
# Explicit 1D heat equation
using Plots
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

    # Create object from struct
    heat = Heat(zeros(n), zeros(n), n, λ, tsteps)
    # Shadow copy for Enzyme
    dheat = Heat(zeros(n), zeros(n), n, λ, tsteps)

    # Boundary conditions
    heat.Tnext[1] = 20.0
    heat.Tnext[end] = 0

    # Compute gradient
    autodiff(
        Enzyme.ReverseWithPrimal,
        sumheat,
        Duplicated(heat, dheat),
        Const(scheme),
        Const(tsteps),
    )

    return heat.Tnext, dheat.Tnext[2:(end-1)]
end
```

Plot function values:
```@example heat
tsteps = 500
T, dT = heat(Revolve(4), tsteps)
```
Plot gradient with respect to sum(T):
```@example heat
plot(dT)
```

# GPU Support

Checkpointing.jl supports GPU arrays from CUDA.jl, AMDGPU.jl, and oneAPI.jl. The checkpointing loop runs on the CPU host, while each iteration can launch GPU kernels internally. GPU support is built on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) for backend detection and [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) for moving data to the GPU.

## Making a struct GPU-compatible

1. **Parameterize the array type** in your struct:
```julia
mutable struct Heat{T}
    Tnext::T
    Tlast::T
    n::Int
    λ::Float64
    tsteps::Int
end
```

2. **Define `Adapt.adapt_structure`** so Adapt.jl can move arrays to the GPU:
```julia
function Adapt.adapt_structure(to, heat::Heat)
    Heat(adapt(to, heat.Tnext), adapt(to, heat.Tlast), heat.n, heat.λ, heat.tsteps)
end
```

3. **Use broadcasting** in loop bodies instead of scalar loops:
```julia
# Broadcasting works on both CPU and GPU arrays
heat.Tnext[2:end-1] .= heat.Tlast[2:end-1] .+ heat.λ .* (
    heat.Tlast[1:end-2] .- 2 .* heat.Tlast[2:end-1] .+ heat.Tlast[3:end]
)
```

## Running on GPU

Create the structs on the CPU and use `adapt` to move them to the GPU:

```julia
using CUDA
using Adapt

n = 100; λ = 0.5; tsteps = 500

# Create on CPU, then adapt to GPU
heat_model = adapt(CuArray, Heat(zeros(n), zeros(n), n, λ, tsteps))
dheat = adapt(CuArray, Heat(zeros(n), zeros(n), n, λ, tsteps))

heat_model.Tnext[1] = 20.0

autodiff(
    Enzyme.ReverseWithPrimal,
    sumheat,
    Duplicated(heat_model, dheat),
    Const(Revolve(4)),
    Const(tsteps),
)
```

## Storage compatibility

| Storage | GPU support |
|---------|-------------|
| `ArrayStorage` (default) | Supported. Checkpoints remain on the GPU device. |
| `HDF5Storage` | Not supported. Throws `ArgumentError` if GPU arrays are detected. |

## GPU memory considerations

Each checkpoint stores a `deepcopy` of the full closure, including all GPU arrays. With `Revolve(c)` using `c` checkpoints, this means `c` copies of all GPU state reside on the device. Monitor GPU memory usage for large models.
