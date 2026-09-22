# Storage

The checkpoint storage types are derived from `AbstractStorage`
```@setup direct_solver
using Checkpointing
```

```julia
using Checkpointing
struct MyStorage{FT} <: AbstractStorage end
```
and have to implement the following functions.

* A constructor taking the number of checkpoints
```julia
MyStorage{FT}(checkpoints::Integer) where {FT}
```
* A `similar` method. A scheme is created before the loop body is known, so it
  starts out parameterized by `Nothing` and is re-created with the concrete
  closure type once the reverse pass begins.
```julia
Base.similar(storage::MyStorage, ::Type{T}) where {T}
```
* Store and restore functions. Both are unexported, so a new backend extends
  `Checkpointing.save!` and `Checkpointing.load!`.
```julia
Checkpointing.save!(storage::MyStorage{FT}, v::FT, i::Integer) where {FT}
Checkpointing.load!(body::FT, storage::MyStorage{FT}, i::Integer) where {FT}
```

`load!` restores checkpoint `i` **into** `body` and returns it, leaving the
stored checkpoint intact -- Revolve restores the same slot more than once.

## Allocation and the copy interface

`save!` and `load!` run once per store and restore action, so a backend that
copies with `deepcopy` allocates a fresh copy of the entire captured state every
time. On a GPU that is one device allocation per action. Backends should instead
build on the two functions below, which split the work into an allocation step
that runs once per slot and a copy step that allocates nothing:

```@docs
Checkpointing.checkpoint_alloc
Checkpointing.checkpoint_copy!
```

`ArrayStorage` fills its slots lazily: the first `save!` to an index allocates
that slot with `checkpoint_alloc`, and every later `save!` to the same index
copies into it with `checkpoint_copy!`. The default `checkpoint_copy!` copies
arrays with `copyto!`, so `CuArray`, `ROCArray`, `oneArray` and `MtlArray` state
is copied device-to-device without leaving the GPU and without scalar indexing.
