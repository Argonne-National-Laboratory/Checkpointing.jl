# Storage

The checkpoint storage types are derived from `AbstractStorage`
```@setup direct_solver
using Checkpointing
```

```julia
using Checkpointing
struct MyStorage{MT} <: AbstractStorage where {MT} end
```
and have to implement the following functions.

* A constructor invoked by the user
```julia
function MyStorage{MT}(n::Int) where {MT} end
```
* save and load functions for the storage type
```julia
Base.load(body::MT, storage::MyStorage{MT}, i::Int) where {MT}
Base.save!(body::MT, storage::MyStorage{MT}, value, i::Int) where {MT}
```
* Size and dimension functions
```julia
Base.size(storage::MyStorage{MT}) where {MT}
Base.ndims(storage::MyStorage{MT}) where {MT}
```