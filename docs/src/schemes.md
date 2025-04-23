# Checkpointing Schemes
A checkpointing scheme may support either for or while loops or both. Each scheme has to define a struct with a constructor derived from the abstract type `Scheme`.
```julia
mutable struct MyScheme{MT} <: Scheme where {MT}
function MyScheme{MT}(...) where {MT}
```
The constructor may take any number of arguments, but the first argument has to be the model type. The model type is the type of the model that is being differentiated.

A scheme has then to implement the reverse pass with the checkpointing scheme that will replace the for or while loop
```julia
function rev_checkpoint_struct_for(
    body::Function,
    alg::Scheme,
    model_input::MT,
    shadowmodel::MT,
    range
) where {MT} end
function rev_checkpoint_struct_while(
    body::Function,
    alg::Scheme,
    model_input::MT,
    shadowmodel::MT,
    condition::Function
) where {MT}
```
The reverse pass for the checkpointing scheme will be called with the loop body `body`, the scheme `alg`, the model input `model_input`, the initialized shadow model `shadowmodel`, and the loop range `range` or the loop condition `condition`.
