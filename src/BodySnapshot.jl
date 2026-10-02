# Snapshots of the loop body of EnzymeCore.checkpoint_for (`EnzymeLLVM`), and
# what of it they hold.
#
# A snapshot is an object of the body's own type. Its *leaves* are the pieces
# of the body a step can change: arrays of bits values, and the bits fields of
# mutable objects; they are numbered depth first, field by field, from the
# body's type alone. A mask says which leaves a snapshot holds (from what
# Enzyme says the step accesses, see `_mask_for`): those are copies, the rest of
# a snapshot is shared with the body. Allocating, storing and restoring a
# snapshot is generated code for the body's type, straight-line field by field,
# with no dynamic dispatch: it compiles with juliac --trim.
#
# Fields of a type whose own leaves cannot be numbered from its type -- of
# abstract type, arrays of references, self-referential types -- are *opaque*
# leaves, copied as `checkpoint_copy!` copies them (dynamically).

# How a field of type `T` is snapshotted: `:bits` (a value, a leaf if its owner
# is mutable), `:array` (an array of bits values, a leaf), `:node` (an object
# whose own fields hold leaves), `:none` (nothing a step can change) or
# `:opaque` (a leaf copied dynamically).
function _snap_kind(@nospecialize(T))
    isbitstype(T) && return :bits
    Base.isbitsunion(T) && return :bits
    isconcretetype(T) || return :opaque
    if T <: Union{Array,Memory}
        return isbitstype(eltype(T)) ? :array : :opaque
    end
    T <: AbstractArray && return :opaque
    fieldcount(T) == 0 && return ismutabletype(T) ? :opaque : :none
    _self_referential(T) && return :opaque
    return :node
end

# Whether an object of type `T` can reach another of type `T`.
function _self_referential(@nospecialize(T))
    seen = Set{Any}()
    work = Any[fieldtype(T, i) for i = 1:fieldcount(T)]
    while !isempty(work)
        F = pop!(work)
        F === T && return true
        (F in seen || !isconcretetype(F) || isbitstype(F)) && continue
        push!(seen, F)
        F <: AbstractArray && (push!(work, eltype(F)); continue)
        append!(work, Any[fieldtype(F, i) for i = 1:fieldcount(F)])
    end
    return false
end

# The number of leaves of an object of type `T`.
function _nleaves(@nospecialize(T))
    n = 0
    for i = 1:fieldcount(T)
        k = _snap_kind(fieldtype(T, i))
        if k === :bits
            n += ismutabletype(T) ? 1 : 0
        elseif k === :array || k === :opaque
            n += 1
        elseif k === :node
            n += _nleaves(fieldtype(T, i))
        end
    end
    return n
end

# The same, a constant of the compiled code.
@generated _nleaves_of(::Type{T}) where {T} = _snap_kind(T) === :node ? _nleaves(T) : 0

# Whether one of the `n` leaves from leaf `b + 1` on is held.
@inline function _any_masked(mask::Vector{Bool}, b::Int, n::Int)
    @inbounds for k = (b+1):(b+n)
        mask[k] && return true
    end
    return false
end

"""
    _snap_alloc(x::T, mask, Val(b)) -> T

A snapshot of `x`, whose leaves are numbered from `b + 1`: the leaves `mask`
holds are copies, objects with no such leaf below them are `x`'s own.
"""
@generated function _snap_alloc(x::T, mask::Vector{Bool}, ::Val{b}) where {T,b}
    _snap_kind(T) === :node || return :(return x)
    args = Any[]
    k = b
    for i = 1:fieldcount(T)
        F = fieldtype(T, i)
        kind = _snap_kind(F)
        if kind === :bits
            push!(args, :(getfield(x, $i)))
            ismutabletype(T) && (k += 1)
        elseif kind === :array
            push!(
                args,
                :(@inbounds(mask[$(k+1)]) ? copy(getfield(x, $i)) : getfield(x, $i)),
            )
            k += 1
        elseif kind === :opaque
            push!(
                args,
                :(@inbounds(mask[$(k+1)]) ? deepcopy(getfield(x, $i)) : getfield(x, $i)),
            )
            k += 1
        elseif kind === :node
            push!(args, :(_snap_alloc(getfield(x, $i), mask, Val($k))))
            k += _nleaves(F)
        else
            push!(args, :(getfield(x, $i)))
        end
    end
    return quote
        _any_masked(mask, $b, $(k - b)) || return x
        return $(Expr(:new, T, args...))
    end
end

"""
    _snap_copy!(dst::T, src::T, mask, Val(b))

Copy the leaves `mask` holds, numbered from `b + 1`, from `src` to `dst`: into a
snapshot to store it, from one to restore it.
"""
@generated function _snap_copy!(dst::T, src::T, mask::Vector{Bool}, ::Val{b}) where {T,b}
    _snap_kind(T) === :node || return :(return nothing)
    body = Any[]
    ismutabletype(T) && push!(body, :(dst === src && return nothing))
    k = b
    for i = 1:fieldcount(T)
        F = fieldtype(T, i)
        kind = _snap_kind(F)
        if kind === :bits
            if ismutabletype(T)
                k += 1
                push!(body, :(@inbounds(mask[$k]) && setfield!(dst, $i, getfield(src, $i))))
            end
        elseif kind === :array
            k += 1
            push!(
                body,
                :(
                    @inbounds(mask[$k]) &&
                    _copy_array!(getfield(dst, $i), getfield(src, $i))
                ),
            )
        elseif kind === :opaque
            k += 1
            push!(body, :(@inbounds(mask[$k]) && _copy_field!(dst, src, Val($i))))
        elseif kind === :node
            push!(body, :(_snap_copy!(getfield(dst, $i), getfield(src, $i), mask, Val($k))))
            k += _nleaves(F)
        end
    end
    push!(body, :(return nothing))
    return Expr(:block, body...)
end

"""
    _snap_bytes(x::T, mask, Val(b)) -> Int

The bytes of the leaves of `x` that `mask` holds.
"""
@generated function _snap_bytes(x::T, mask::Vector{Bool}, ::Val{b}) where {T,b}
    _snap_kind(T) === :node || return :(return 0)
    terms = Any[]
    k = b
    for i = 1:fieldcount(T)
        F = fieldtype(T, i)
        kind = _snap_kind(F)
        if kind === :bits
            if ismutabletype(T)
                k += 1
                push!(terms, :(@inbounds(mask[$k]) ? $(sizeof(F)) : 0))
            end
        elseif kind === :array
            k += 1
            push!(terms, :(@inbounds(mask[$k]) ? sizeof(getfield(x, $i)) : 0))
        elseif kind === :opaque
            # Its own memory only: what it refers to is not counted.
            k += 1
            push!(terms, :(@inbounds(mask[$k]) ? Core.sizeof(getfield(x, $i)) : 0))
        elseif kind === :node
            push!(terms, :(_snap_bytes(getfield(x, $i), mask, Val($k))))
            k += _nleaves(F)
        end
    end
    return Expr(:call, :+, 0, terms...)
end

# Where a leaf is in the memory Enzyme describes accesses to (`set_paths`): the
# byte offsets of the pointer fields followed from the box of the body to the
# object holding it, and the bytes of its field in that object. `ref`: the
# field points to the leaf (an array, an opaque value).
struct LeafPos
    hops::Vector{Int}
    lo::Int
    hi::Int
    ref::Bool
end

function _leaf_positions!(out, @nospecialize(T), hops::Vector{Int}, base::Int)
    for i = 1:fieldcount(T)
        F = fieldtype(T, i)
        kind = _snap_kind(F)
        off = base + Int(fieldoffset(T, i))
        inl = Base.allocatedinline(F)
        size = inl ? Int(sizeof(F)) : sizeof(Ptr{Cvoid})
        if kind === :bits
            ismutabletype(T) && push!(out, LeafPos(hops, off, off + max(size, 1), false))
        elseif kind === :array || kind === :opaque
            push!(out, LeafPos(hops, off, off + size, !inl))
        elseif kind === :node
            if inl
                _leaf_positions!(out, F, hops, off)
            else
                _leaf_positions!(out, F, [hops; off], 0)
            end
        end
    end
    return out
end

# The positions of the leaves of a body of type `B`, in leaf order, for its box
# (`Base.RefValue{B}`): the box holds the body inline or points to it.
@generated function _leaf_table(::Type{B}) where {B}
    _snap_kind(B) === :node || return :(LeafPos[])
    R = Base.RefValue{B}
    hops = Base.allocatedinline(B) ? Int[] : Int[Int(fieldoffset(R, 1))]
    base = Base.allocatedinline(B) ? Int(fieldoffset(R, 1)) : 0
    table = _leaf_positions!(LeafPos[], B, hops, base)
    @assert length(table) == _nleaves(B)
    return Expr(
        :vect,
        (
            :(LeafPos($(Expr(:vect, p.hops...)), $(p.lo), $(p.hi), $(p.ref))) for p in table
        )...,
    )
end

_startswith(a::Vector{Int}, p::Vector{Int}) =
    length(a) >= length(p) && view(a, 1:length(p)) == p

# Whether some leaf is in or below the object at `prefix`.
function _any_under(table::Vector{LeafPos}, prefix::Vector{Int})
    for leaf in table
        _startswith(leaf.hops, prefix) && return true
    end
    return false
end

"""
    _mask_for(table, paths, snapshot) -> Vector{Bool}

The leaves a snapshot holds: those the step writes, and with `snapshot =
:accessed` also those it reads. An access Enzyme describes no further than an
object (the whole object, or through a pointer that is not a leaf's) holds all
the leaves in or below that object.
"""
function _mask_for(table::Vector{LeafPos}, paths::Vector{AccessPath}, snapshot::Symbol)
    mask = fill(false, length(table))
    for access in paths
        keep = access.write || (snapshot === :accessed && access.read)
        # The box itself only ever holds the body.
        (keep && !isempty(access.path)) || continue
        P = access.path
        o = access.offset
        hit = false
        for (k, leaf) in enumerate(table)
            if leaf.hops == P && o >= 0 && leaf.lo <= o < leaf.hi
                # The access is to the leaf's field.
                mask[k] = hit = true
            elseif leaf.ref &&
                   length(P) > length(leaf.hops) &&
                   _startswith(P, leaf.hops) &&
                   leaf.lo <= P[length(leaf.hops)+1] < leaf.hi
                # It is to what the leaf's field points to.
                mask[k] = hit = true
            end
        end
        hit && continue
        # The whole object at the end of the path, or one no leaf describes:
        # everything in or below the deepest object on the path that leaves
        # are in or below.
        n = length(P)
        while n > 0 && !_any_under(table, P[1:n])
            n -= 1
        end
        prefix = P[1:n]
        for (k, leaf) in enumerate(table)
            _startswith(leaf.hops, prefix) && (mask[k] = true)
        end
    end
    return mask
end
