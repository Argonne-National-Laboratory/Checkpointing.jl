"""
    ArrayStorage

Array (RAM) storage for checkpointing.

"""
struct ArrayStorage{FT} <: AbstractStorage where {FT}
    _fstorage::Array{FT}
end

function Base.similar(storage::ArrayStorage{MT}, ::Type{T}) where {MT,T}
    ArrayStorage{T}(size(storage._fstorage, 1))
end

function ArrayStorage{FT}(acp::Int64) where {FT}
    fstorage = Array{FT}(undef, acp)
    return ArrayStorage(fstorage)
end

function save!(storage::ArrayStorage{FT}, v::FT, i::Int64) where {FT}
    storage._fstorage[i] = v
end

Base.ndims(::Type{ArrayStorage{FT}}) where {FT} = 1
Base.size(storage::ArrayStorage{FT}) where {FT} = size(storage._storage)

function load(body::MT, storage::ArrayStorage{MT}, i::Int64) where {MT}
    return storage._fstorage[i]
end