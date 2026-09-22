"""
    HDF5Storage{MT}(checkpoints::Integer; filename = tempname())

A storage type for checkpointing that uses HDF5 files to store the checkpoints.

"""
mutable struct HDF5Storage{MT} <: AbstractStorage
    fid::HDF5.File
    filename::String
    acp::Int64
end

function HDF5Storage{Nothing}(acp::Integer; filename = tempname())
    fid = h5open(filename, "w")
    storage = HDF5Storage{Nothing}(fid, filename, acp)
    close(fid)
    return storage
end

function HDF5Storage{MT}(acp::Integer; filename = tempname()) where {MT}
    fid = h5open(filename, "w")
    storage = HDF5Storage{MT}(fid, filename, acp)
    function _finalizer(storage::HDF5Storage{MT})
        close(storage.fid)
        return storage
    end
    finalizer(_finalizer, storage)
    return storage
end

function Base.similar(storage::HDF5Storage, ::Type{T}) where {T}
    # A prototype can be instantiated more than once -- nested schemes share one
    # prototype object, and every reverse sweep instantiates again -- so each
    # instantiation needs its own file. Reopening `storage.filename` in "w" mode
    # would truncate the sibling scheme's checkpoints out from under it.
    return HDF5Storage{T}(storage.acp; filename = _sibling_filename(storage.filename))
end

function _sibling_filename(base::AbstractString)
    dir, file = splitdir(base)
    root, ext = splitext(file)
    return joinpath(dir, string(root, "_", string(rand(UInt32); base = 16), ext))
end

function Base.getindex(storage::HDF5Storage{MT}, i)::MT where {MT}
    @assert i >= 1 && i <= storage.acp
    blob = read(storage.fid["$i"])
    return deserialize(blob)
end

function save!(storage::HDF5Storage{MT}, v::MT, i::Integer) where {MT}
    @assert i >= 1 && i <= storage.acp
    if haskey(storage.fid, "$i")
        delete_object(storage.fid, "$i")
    end
    blob = serialize(v)
    storage.fid["$i"] = blob
    return storage
end

"""
    load!(body, storage, i) -> body

Restore checkpoint `i` into `body` in place and return it.
"""
function load!(body::MT, storage::HDF5Storage{MT}, i::Integer) where {MT}
    @assert i >= 1 && i <= storage.acp
    blob = read(storage.fid["$i"])
    return checkpoint_copy!(body, deserialize(blob))
end
