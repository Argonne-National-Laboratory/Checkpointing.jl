"""
    ArrayStorage

Array (RAM) storage for checkpointing.

"""
struct ArrayStorage{FT} <: AbstractStorage where {FT}
    _fstorage::Array{FT}
end

function ArrayStorage{FT}(acp::Int) where {FT}
    fstorage = Array{FT}(undef, acp)
    return ArrayStorage(fstorage)
end

function Base.getindex(storage::ArrayStorage{FT}, i) where {FT}
    storage._fstorage[i]
end

function Base.setindex!(storage::ArrayStorage{FT}, v, i) where {FT}
    storage._fstorage[i] = v
end

Base.ndims(::Type{ArrayStorage{FT}}) where {FT} = 1
Base.size(storage::ArrayStorage{FT}) where {FT} = size(storage._storage)
