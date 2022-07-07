mutable struct HDF5Storage{MT} <: AbstractStorage where {MT}
    fid::HDF5.File
    filename::String
    acp::Int64
end

function HDF5Storage{MT}(acp::Int; filename=tempname()) where {MT}
    fid = h5open(filename, "w")
    storage = HDF5Storage{MT}(fid, filename, acp)
    function _finalizer(storage::HDF5Storage{MT})
        close(storage.fid)
        return storage
    end
    finalizer(_finalizer, storage)
    return storage
end

function Base.getindex(storage::HDF5Storage{MT}, i)::MT where {MT}
    @assert i >= 1 && i <= storage.acp
    blob = read(storage.fid["$i"])
    return deserialize(blob)
end

function Base.setindex!(storage::HDF5Storage{MT}, v::MT, i) where {MT}
    @assert i >= 1 && i <= storage.acp
    if haskey(storage.fid, "$i")
        delete_object(storage.fid, "$i")
    end
    blob = serialize(v)
    storage.fid["$i"] = blob
end
