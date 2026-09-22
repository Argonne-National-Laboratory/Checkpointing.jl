"""
    ArrayStorage{FT}(checkpoints::Integer)

Array (RAM) storage for checkpointing.

Slots are filled lazily: the first `save!` to a given index allocates that slot
with [`checkpoint_alloc`](@ref), and every later `save!` to the same index
copies into it with [`checkpoint_copy!`](@ref). A reverse sweep therefore
allocates at most `checkpoints` copies of the state no matter how many store and
restore actions the scheme performs, which is what keeps the sweep allocation-
free once it is warm -- on the GPU as much as on the host.
"""
struct ArrayStorage{FT} <: AbstractStorage
    _fstorage::Vector{FT}
end

function ArrayStorage{FT}(acp::Integer) where {FT}
    return ArrayStorage{FT}(Vector{FT}(undef, acp))
end

function Base.similar(storage::ArrayStorage, ::Type{T}) where {T}
    return ArrayStorage{T}(length(storage._fstorage))
end

Base.ndims(::Type{<:ArrayStorage}) = 1
Base.size(storage::ArrayStorage) = size(storage._fstorage)
Base.length(storage::ArrayStorage) = length(storage._fstorage)

function save!(storage::ArrayStorage{FT}, v::FT, i::Integer) where {FT}
    checkbounds(storage._fstorage, i)
    if isassigned(storage._fstorage, i)
        @inbounds checkpoint_copy!(storage._fstorage[i], v)
    else
        @inbounds storage._fstorage[i] = checkpoint_alloc(v)
    end
    return storage
end

"""
    load!(body, storage, i) -> body

Restore checkpoint `i` into `body` in place and return it.

The checkpoint itself is left untouched, so the same slot can be restored more
than once -- which Revolve relies on.
"""
function load!(body::FT, storage::ArrayStorage{FT}, i::Integer) where {FT}
    checkbounds(storage._fstorage, i)
    isassigned(storage._fstorage, i) || throw(
        ArgumentError(
            "[Checkpointing.jl]: checkpoint $i was restored before it was stored.",
        ),
    )
    return @inbounds checkpoint_copy!(body, storage._fstorage[i])
end
