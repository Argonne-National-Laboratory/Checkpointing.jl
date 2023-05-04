# Checkpointing
[Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl) provides checkpointing schemes for adjoint computations using automatic differentiation (AD) of time stepping loops. Currently, we support the macro @checkpoint_struct, which differentiates and checkpoints a struct used in the loop. Each loop iteration is differentiated using Enzyme.jl. We rely on ChainRulesCore.jl to integrate with AD tools applied to the code outside of the loop.

The schemes are agnostic to the AD tool being used and can be easily interfaced with any Julia AD tool. Currently, the package provides the following checkpointing schemes:

* Online checkpointing schemes for adaptive timestepping
* Revolve/Binomial checkpointing [1]
* Periodic checkpointing

## Future
The following features are planned for development:

* Composition of checkpointing schemes
* Multi-level checkpointing schemes

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