# Checkpointing

[Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl) provides checkpointing schemes for adjoint computations using automatic differentiation (AD) of time-stepping loops. Currently, we support the macro `@ad_checkpoint`, which differentiates and checkpoints a struct used in a while or for the loop with a `UnitRange`.

Each loop iteration is differentiated using [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl). We rely on external differentiation rule systems to integrate with AD tools applied to the code outside of the loop.

The schemes are agnostic to the AD tool being used and can be easily interfaced with any Julia AD tool. Currently, the package supports:

## Scheme
* Revolve/Binomial checkpointing [1]
* Periodic checkpointing
* Online r=2 checkpointing for a while loops with a priori unknown number of iterations [2]

## Rules
* [EnzymeRules.jl](https://enzyme.mit.edu/julia/stable/generated/custom_rule/)

## Storage
* ArrayStorage: Stores all checkpoints values in an array of type `Array` (supports GPU arrays)
* HDF5Storage: Stores all checkpoints values in an HDF5 file (CPU only)

## GPU Support
Checkpointing.jl supports GPU arrays from CUDA.jl, AMDGPU.jl, and oneAPI.jl via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl). Parameterize your structs with the array type, define `Adapt.adapt_structure`, and use broadcasting in loop bodies. See the [Quick Start](@ref) guide for a GPU example.

## Limitations
* Currently, the package only supports `UnitRange` ranges in `for` loops. We will add range types on a per-need basis. Please, open an issue if you need support for a specific range type.
* We only support Enzyme as the differentiation tool of the loop body. This is due to our strict requirement for a mutation-enabled AD tool in our projects. However, there is no fundamental reason why we could not support other AD tools. Please, open an issue if you need support for a specific AD tool.
* We don't support any activity analysis. This implies that loop iterators have to be part of the checkpointed struct if they are used in the loop body. Currently, we store the entire struct at each checkpoint. This is not necessary, and we will add support for storing only the required fields in the future.
## Quick Start



```@contents
Pages = [
    "quickstart.md",
]
Depth=1
```
## API
```@contents
Pages = [
    "lib/checkpointing.md",
]
Depth = 1
```
## References
[1] Andreas Griewank and Andrea Walther. 2000. Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation. ACM Trans. Math. Softw. 26, 1 (March 2000), 19–45. DOI:https://doi.org/10.1145/347837.347846