# Checkpointing
[Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl)
provides checkpointing schemes, storage methods, and
differentiation rules interfaces for adjoint computations using automatic
differentiation (AD) of time-stepping loops. Currently, we support the macro
`@checkpoint_struct`, which differentiates and checkpoints a struct used in a
for or while loop. Each loop iteration is differentiated using
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl). We rely on
[ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) and
[EnzymeRules.jl](https://enzymead.github.io/Enzyme.jl/dev/generated/custom_rule/)
to integrate with AD tools applied to the code outside of the loop.

The following checkpointing schemes are currently supported:

1. Revolve/Binomial checkpointing [1],
2. Periodic checkpointing,
3. Online r=2 checkpointing for a while loop with a priori unknown number of iterations [2].

In addition, the user has the choice of two storage types for the checkpoints:

1. `ArrayStorage`: Stores the checkpoints in an array,
2. `HDF5Storage`: Stores the checkpoints in an HDF5 for large checkpoints on disk.

## Future

The following features are planned for development:

* Composition of checkpointing schemes,
* Multi-level checkpointing schemes,
* Storage for GPU memory checkpointing.

## Quick Start

```@contents
Pages = [
    "quickstart.md",
]
Depth=1
```
## Library
```@contents
Pages = [
    "lib/checkpointing.md",
]
Depth = 1
```
## References
[1] Andreas Griewank and Andrea Walther. 2000. Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation. ACM Trans. Math. Softw. 26, 1 (March 2000), 19–45. DOI:https://doi.org/10.1145/347837.347846