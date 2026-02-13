```@meta
CurrentModule = Checkpointing
```
# API

## Decorator macros for loops
```@docs
@ad_checkpoint

```

## Supported Schemes
```@docs
Revolve
Periodic
Online_r2

```

## Supported Storages
```@docs
ArrayStorage
HDF5Storage

```

## GPU Utilities
```@docs
is_gpu_array
check_no_gpu_arrays

```

## Developer variables for implementing new schemes
```@docs
Scheme

```