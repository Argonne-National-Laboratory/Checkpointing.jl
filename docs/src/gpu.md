# GPU

Checkpointing.jl needs no GPU-specific code or dependency. The checkpointing
schedule runs on the host; each iteration of the loop body launches whatever
kernels it launches. Checkpoints of device arrays stay on the device: storage
slots are allocated once with [`Checkpointing.checkpoint_alloc`](@ref) and
refilled with `copyto!` through [`Checkpointing.checkpoint_copy!`](@ref), so a
store or restore is one device-to-device copy with no allocation and no
transfer to the host. This works for any array type that implements `copyto!` --
`CuArray`, `ROCArray`, `oneArray`, `MtlArray`.

`examples/heat_gpu.jl` runs the heat equation of the quick start on either
the host or a device:

```julia
using CUDA
include("examples/heat_gpu.jl")
T, dT = heat_device(Revolve(100), 500; arraytype = CuArray)
```

and gives the same result as on the host.

## Writing a loop body Enzyme can differentiate on a device

These constraints come from differentiating GPU broadcasts with Enzyme, not from
Checkpointing.jl, and apply equally without `@ad_checkpoint`:

- **No floating-point scalars inside a broadcast.** `x .= λ .* y` with a
  `Float64` `λ` -- a struct field, or even a literal `0.5` -- fails with
  `MethodError: no method matching MixedDuplicated(::Broadcasted...)`. Keep
  coefficients in device arrays. Integer literals are fine.
- **No views.** Broadcasting over `@views` slices of a device array aborts
  Enzyme with `number of arg operands != function parameters`. Use whole-array
  broadcasts; `circshift` expresses stencil neighbours.
- **No scalar indexing**, which Enzyme cannot differentiate on a device.

`examples/heat_gpu.jl` shows all three: the diffusion coefficient is a device
array that is zero at both ends, which keeps the boundary cells fixed and
cancels the wraparound terms `circshift` introduces.

## Performance

A GPU only pays off once the state is large enough to amortise kernel launches.
Revolve with 20 checkpoints over 200 steps, on an RTX 4080 against one host
thread:

| cells | CPU | GPU | speedup |
|---:|---:|---:|---:|
| 10³ | 0.003 s | 0.235 s | 0.01× |
| 10⁵ | 0.374 s | 0.233 s | 1.6× |
| 10⁶ | 6.14 s | 0.241 s | 25× |
| 10⁷ | 98.0 s | 2.22 s | 44× |

The largest size varies by roughly ±15% between runs.

Reproduce with `julia --project=test/gpu test/gpu/benchmark.jl`.

## Running the GPU tests

The GPU tests have their own environment, so the CPU test suite never installs
CUDA:

```
julia --project=test/gpu -e 'using Pkg; Pkg.instantiate()'
julia --project=test/gpu test/gpu/runtests.jl
```
