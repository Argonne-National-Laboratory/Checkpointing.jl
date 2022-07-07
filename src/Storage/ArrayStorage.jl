struct ArrayStorage{MT} <: AbstractStorage where {MT}
    _storage::Array{MT}
end

function ArrayStorage{MT}(acp::Int) where {MT}
    storage = Array{MT}(undef, acp)
    return ArrayStorage(storage)
end

Base.getindex(storage::ArrayStorage{MT}, i) where {MT} = storage._storage[i]

function Base.setindex!(storage::ArrayStorage{MT}, v, i) where {MT}
    storage._storage[i] = v
end

Base.ndims(::Type{ArrayStorage{MT}}) where {MT} = 1
Base.size(storage::ArrayStorage{MT}) where {MT} = size(storage._storage)
