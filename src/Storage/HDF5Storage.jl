using Accessors

"""
    HDF5Storage

A storage type for checkpointing that uses HDF5 files to store the checkpoints.

"""
mutable struct HDF5Storage{MT} <: AbstractStorage where {MT}
    fid::HDF5.File
    filename::String
    acp::Int64
end

function HDF5Storage{Nothing}(acp::Int; filename = tempname())
    fid = h5open(filename, "w")
    storage = HDF5Storage{Nothing}(fid, filename, acp)
    close(fid)
    return storage
end

function HDF5Storage{MT}(acp::Int; filename = tempname()) where {MT}
    fid = h5open(filename, "w")
    storage = HDF5Storage{MT}(fid, filename, acp)
    function _finalizer(storage::HDF5Storage{MT})
        close(storage.fid)
        return storage
    end
    finalizer(_finalizer, storage)
    return storage
end

function Base.similar(storage::HDF5Storage{MT}, ::Type{T}) where {MT,T}
    HDF5Storage{T}(storage.acp; filename = storage.filename)
end

function Base.getindex(storage::HDF5Storage{MT}, i)::MT where {MT}
    @assert i >= 1 && i <= storage.acp
    blob = read(storage.fid["$i"])
    return deserialize(blob)
end

function save!(storage::HDF5Storage{MT}, v::MT, i::Int64) where {MT}
    @assert i >= 1 && i <= storage.acp
    if isa(v, Function)
        check_no_gpu_arrays(v)
    end
    if haskey(storage.fid, "$i")
        delete_object(storage.fid, "$i")
    end
    blob = serialize(v)
    storage.fid["$i"] = blob
end

@generated function hdf5_update!(dest::MT1, src::MT2) where {MT1,MT2}
    # assignments = [:(@reset :(dest.$name = src.$name)) for name in fieldnames(MT1)]
    assignments = [
        Expr(
            :macrocall,
            Symbol("@reset"),
            LineNumberNode(@__LINE__),
            :(dest.$name = src.$name),
        ) for name in fieldnames(MT1)
    ]
    quote
        $(assignments...)
    end
end

function load(body::MT, storage::HDF5Storage{MT}, i::Int64) where {MT}
    @assert i >= 1 && i <= storage.acp
    blob = read(storage.fid["$i"])
    _body = deserialize(blob)
    hdf5_update!(body, _body)
end
