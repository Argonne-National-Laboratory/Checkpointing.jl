# The copy interface that every storage backend is built on.
#
# Checkpointing a time-stepping loop means snapshotting the loop body's closure
# once per `store` action and restoring it once per `restore` action. Doing that
# with `deepcopy` allocates a fresh copy of the entire captured state every time,
# which on a GPU means one device allocation per store *and* per restore. The
# two functions below split that into an allocation step that runs once per
# checkpoint slot and a copy step that runs in the reverse sweep and allocates
# nothing.

"""
    checkpoint_alloc(x)

Allocate a checkpoint slot able to hold a snapshot of `x`.

This runs once per checkpoint slot and never inside the reverse sweep, so it is
free to allocate. The default is a `deepcopy`, which gives the slot the same
field and aliasing structure as `x`; [`checkpoint_copy!`](@ref) then reuses that
structure on every subsequent store.

Storage backends that need control over where a slot lives -- pinned host
memory, a device buffer, an offload pool -- specialize this.
"""
checkpoint_alloc(x) = deepcopy(x)

"""
    checkpoint_copy!(dst, src) -> dst

Overwrite `dst` with a snapshot of `src`, in place.

`dst` must have come from [`checkpoint_alloc`](@ref) applied to a value of the
same type, so that the two already agree on which objects are shared. Arrays are
copied with `copyto!`, which resolves to a device-to-device copy for GPU arrays;
that is what keeps the reverse sweep free of allocation on host and device
alike.

After the call `dst` must be unaffected by later mutation of `src`, so anything
that cannot be written through in place is replaced with a fresh `deepcopy`.
"""
function checkpoint_copy! end

# --- arrays ---------------------------------------------------------------

# `dst` and `src` may differ in type when a backend offloads between memory
# spaces (a device checkpoint restored into a host slot, say); `copyto!` already
# knows how to bridge those.
@noinline checkpoint_copy!(dst::AbstractArray, src::AbstractArray) = _copy_array!(dst, src)

# Disambiguates against the struct method below for `T <: AbstractArray`.
@noinline checkpoint_copy!(dst::T, src::T) where {T<:AbstractArray} = _copy_array!(dst, src)

function _copy_array!(dst::AbstractArray, src::AbstractArray)
    dst === src && return dst
    axes(dst) == axes(src) || _resize_like!(dst, src)
    if isbitstype(eltype(dst)) && eltype(dst) === eltype(src)
        # The fast path: one memcpy on the host, one device-to-device copy on a
        # GPU, and no scalar indexing either way.
        copyto!(dst, src)
    else
        for i in eachindex(dst, src)
            _copy_element!(dst, src, i)
        end
    end
    return dst
end

_resize_like!(dst::Vector, src::AbstractVector) = (resize!(dst, length(src)); dst)

function _resize_like!(dst::AbstractArray, src::AbstractArray)
    throw(
        ArgumentError(
            "[Checkpointing.jl]: checkpoint slot of type $(typeof(dst)) has axes $(axes(dst)) " *
            "but the state being stored has axes $(axes(src)). Only `Vector` slots can be " *
            "resized, so the shape of checkpointed state must stay fixed across iterations.",
        ),
    )
end

@inline function _copy_element!(dst, src, i)
    isassigned(src, i) || return nothing
    s = src[i]
    if isbits(s)
        dst[i] = s
    elseif isassigned(dst, i) && _writable_through(dst[i], s)
        checkpoint_copy!(dst[i], s)
    else
        dst[i] = deepcopy(s)
    end
    return nothing
end

# --- structs --------------------------------------------------------------

@noinline checkpoint_copy!(dst::T, src::T) where {T} = _copy_fields!(dst, src)

# `dst` and `src` are usually the same type, but not always: `Serialization`
# rebuilds an anonymous closure under `Serialization.__deserialized_types__` and
# mints a *brand new type every time it deserializes*. Specializing on those
# would recompile the copy for every checkpoint `HDF5Storage` restores, so the
# cross-type path is deliberately left dynamic. It runs once per restore, at the
# top of the object graph; everything below it is nominally typed and takes the
# generated fast path above.
function checkpoint_copy!(@nospecialize(dst), @nospecialize(src))
    return _copy_fields_dynamic!(dst, src)
end

function _copy_fields_dynamic!(@nospecialize(dst), @nospecialize(src))
    Td, Ts = typeof(dst), typeof(src)
    (isbitstype(Td) || fieldcount(Td) == 0) && return dst
    fieldnames(Td) == fieldnames(Ts) || _field_mismatch(dst, src)
    mut = ismutabletype(Td)
    for i = 1:fieldcount(Td)
        isdefined(src, i) || continue
        s = getfield(src, i)
        if isbits(s)
            mut && setfield!(dst, i, s)
            continue
        end
        if isdefined(dst, i)
            d = getfield(dst, i)
            if _writable_through(d, s)
                checkpoint_copy!(d, s)
                continue
            end
        end
        mut && setfield!(dst, i, deepcopy(s))
    end
    return dst
end

# Unrolled so that each `getfield` has a concrete type; the generic
# `for i in 1:fieldcount(T)` version infers `Any` and allocates on every field.
@generated function _copy_fields!(dst::T, src::T) where {T}
    # Nothing reachable from an isbits value can be mutated, so whoever owns
    # this field has already stored it by value.
    (isbitstype(T) || fieldcount(T) == 0) && return :(return dst)

    mut = ismutabletype(T)
    exprs = Expr[:(dst === src && return dst)]
    for i = 1:fieldcount(T)
        if isbitstype(fieldtype(T, i))
            # An isbits field of an immutable parent is fixed for the lifetime
            # of the object, so there is nothing to do.
            mut && push!(exprs, :(setfield!(dst, $i, getfield(src, $i))))
        else
            push!(exprs, :(_copy_field!(dst, src, Val($i))))
        end
    end
    push!(exprs, :(return dst))
    return Expr(:block, exprs...)
end

@noinline function _field_mismatch(dst, src)
    throw(
        ArgumentError(
            "[Checkpointing.jl]: cannot restore a checkpoint of type $(typeof(src)) into " *
            "$(typeof(dst)): their fields are $(fieldnames(typeof(src))) and " *
            "$(fieldnames(typeof(dst))). Checkpointed state must keep the same shape " *
            "across iterations.",
        ),
    )
end

@inline function _copy_field!(dst::T, src::T, ::Val{i}) where {T,i}
    isdefined(src, i) || return nothing
    s = getfield(src, i)
    if isbits(s)
        ismutabletype(T) && setfield!(dst, i, s)
        return nothing
    end
    if isdefined(dst, i)
        d = getfield(dst, i)
        if _writable_through(d, s)
            checkpoint_copy!(d, s)
            return nothing
        end
    end
    # Either the slot holds a different shape than the state now does, or the
    # value is an immutable leaf (a `String`, say) that can only be replaced.
    ismutabletype(T) && setfield!(dst, i, deepcopy(s))
    # For an immutable parent there is nothing to replace: its fields are fixed
    # for the lifetime of the object, so the slot already holds the right value.
    return nothing
end

# Can `d` be overwritten in place with the contents of `s`? Only values that own
# storage we can write through: mutable objects, and immutable containers
# (tuples, immutable structs) whose own fields may be mutable. Immutable leaves
# with no fields -- `String`, `Symbol` -- must be replaced instead.
@inline function _writable_through(d, s)
    Td, Ts = typeof(d), typeof(s)
    (Td === Ts || fieldnames(Td) == fieldnames(Ts)) || return false
    return ismutable(d) || fieldcount(Td) > 0
end
