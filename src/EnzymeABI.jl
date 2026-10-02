# The checkpointing interface of Enzyme's LLVM core (enzyme/checkpoint.h).
#
# Enzyme differentiates `__enzyme_checkpoint_for(step, start, n, enzyme_scheme,
# scheme, data, ...)` -- in C, C++, Fortran or any other language compiled with
# Enzyme -- by asking a scheme for actions through a table of C function
# pointers. This file provides that table for the schemes of this package, so
# they schedule loops that Enzyme differentiates, and their storage keeps the
# snapshots. Enzyme decides what a snapshot holds and hands it over as a list
# of memory regions; here it is packed into a byte vector and stored.
#
# The actions are this package's `Action`: the flag values of `ActionFlag` and
# the layout of `Action` are those of the C interface.

"""
    EnzymeCkptRegion

A piece of memory a snapshot holds (`EnzymeCkptRegion` in enzyme/checkpoint.h).
"""
struct EnzymeCkptRegion
    ptr::Ptr{UInt8}
    bytes::UInt64
    addrspace::UInt32
    flags::UInt32
end

"""
    EnzymeCheckpointScheme

The table of callbacks Enzyme drives a scheme through
(`EnzymeCheckpointScheme` in enzyme/checkpoint.h).
"""
struct EnzymeCheckpointScheme
    version::UInt32
    init::Ptr{Cvoid}
    next_action::Ptr{Cvoid}
    store::Ptr{Cvoid}
    restore::Ptr{Cvoid}
    set_nsteps::Ptr{Cvoid}
    finalize::Ptr{Cvoid}
    save_state::Ptr{Cvoid}
    load_state::Ptr{Cvoid}
    set_paths::Ptr{Cvoid}
end

const ENZYME_CKPT_ABI_VERSION = UInt32(1)

"""
    PeriodicActions

The schedule of [`Periodic`](@ref) as actions: the forward sweep stores the
start of every segment; reversing a segment restores its start and stores the
state before each of its steps. Slots `0:K-1` hold segment starts, slots `K:`
the steps of the segment being reversed. `Periodic` itself runs its schedule
directly and has no `next_action!`.
"""
mutable struct PeriodicActions
    steps::Int
    segments::Int
    queue::Vector{Action}
    pos::Int
    segment::Int
end

_segstart(p::PeriodicActions, k) = div(k * p.steps, p.segments)

# The slot of the state before step j of segment k: the segment's own slot k
# for its first step, then slots K on, shared by all segments.
_segslot(p::PeriodicActions, k, j) =
    j == _segstart(p, k) ? k : p.segments + (j - _segstart(p, k) - 1)

# Every segment but the last already has its start in slot k, from the forward
# sweep.
function _queue_segment!(p::PeriodicActions, k, first::Bool)
    s, e = _segstart(p, k), _segstart(p, k + 1)
    for j = s:(e-2)
        (j != s || first) && push!(p.queue, Action(store, j, j, _segslot(p, k, j)))
        push!(p.queue, Action(forward, j + 1, j, _segslot(p, k, j)))
    end
    push!(p.queue, Action(first ? firstuturn : uturn, e, e - 1, _segslot(p, k, e - 1)))
    for j = (e-2):-1:s
        push!(p.queue, Action(restore, j, j, _segslot(p, k, j)))
        push!(p.queue, Action(uturn, j + 1, j, _segslot(p, k, j)))
    end
    return p
end

function PeriodicActions(steps::Int, segments::Int)
    segments = max(1, min(segments, steps))
    p = PeriodicActions(steps, segments, Action[], 0, segments - 1)
    if steps == 0
        push!(p.queue, Action(done, 0, 0, -1))
        return p
    end
    for k = 0:(segments-2)
        push!(p.queue, Action(store, _segstart(p, k), _segstart(p, k), k))
        push!(p.queue, Action(forward, _segstart(p, k + 1), _segstart(p, k), k))
    end
    _queue_segment!(p, p.segment, true)
    return p
end

function next_action!(p::PeriodicActions)::Action
    if p.pos == length(p.queue)
        empty!(p.queue)
        p.pos = 0
        if p.segment == 0 || p.steps == 0
            push!(p.queue, Action(done, 0, 0, -1))
        else
            p.segment -= 1
            k = p.segment
            push!(p.queue, Action(restore, _segstart(p, k), _segstart(p, k), k))
            _queue_segment!(p, k, false)
        end
    end
    p.pos += 1
    return p.queue[p.pos]
end

"""
    AccessPath

An access of a checkpointed step, from Enzyme's `set_paths`: the byte offsets of
the pointer fields followed from the loop body, the byte offset of the access in
the object reached (`-1` for the whole object), and whether it reads or writes.
"""
struct AccessPath
    path::Vector{Int}
    offset::Int
    read::Bool
    write::Bool
end

function decode_paths(paths::Ptr{Int64}, len::Integer)
    out = AccessPath[]
    k = 1
    while k <= len
        n = unsafe_load(paths, k)
        path = [Int(unsafe_load(paths, k + j)) for j = 1:n]
        offset = Int(unsafe_load(paths, k + n + 1))
        flags = unsafe_load(paths, k + n + 2)
        push!(out, AccessPath(path, offset, flags & 1 != 0, flags & 2 != 0))
        k += n + 3
    end
    return out
end

_inline(T) = Base.allocatedinline(T)

# The field of `T` at byte `offset`, looking into fields stored inline: its
# field path from `T`, its type, and the offset left over inside it.
function _field_at(T::DataType, offset::Int)
    isstructtype(T) || return nothing
    for i = 1:fieldcount(T)
        fo = Int(fieldoffset(T, i))
        ft = fieldtype(T, i)
        size = _inline(ft) ? sizeof(ft) : sizeof(Ptr{Cvoid})
        fo <= offset < fo + max(size, 1) || continue
        if _inline(ft) && ft isa DataType && isstructtype(ft) && fieldcount(ft) > 0
            inner = _field_at(ft, offset - fo)
            inner === nothing && return ((i,), ft, offset - fo)
            return ((i, inner[1]...), inner[2], inner[3])
        end
        return ((i,), ft, offset - fo)
    end
    return nothing
end

_getfields(obj, fields) = foldl(getfield, fields; init = obj)

# The fields leading to the object the pointer field at byte `offset` of an
# object of type `T` points to, or nothing if that is not a Julia object
# reference (an array's data, say).
function _deref_fields(T::DataType, offset::Int)
    T <: AbstractArray && return nothing
    field = _field_at(T, offset)
    field === nothing && return nothing
    fields, ft, rest = field
    (rest == 0 && !_inline(ft)) || return nothing
    return fields
end

"""
    MaskedSnapshot

What a snapshot of the loop body holds when Enzyme said what the step accesses:
the objects it writes (or reads) as a whole -- typically arrays -- and the
scalar fields of mutable objects.
"""
mutable struct MaskedSnapshot
    objects::Vector{Any}
    scalars::Vector{Any}
end

# How to find one piece of the loop body a snapshot holds: the fields to follow
# from the box, each after checking the type of the object it is taken from,
# and at the end the whole object (field 0) or a scalar field of a mutable one
# of type `owner`.
struct Leaf
    hops::Vector{Tuple{DataType,Vector{Int}}}
    owner::DataType
    field::Int
end

# The leaves of the snapshot, worked out from the types of the objects met in
# `box`: whole objects, and scalar fields of mutable objects.
function _plan_leaves(sched, box)
    leaves = Leaf[]
    for access in sched.paths
        keep = access.write || (sched.snapshot === :accessed && access.read)
        # The box itself only ever holds the body.
        (keep && !isempty(access.path)) || continue
        obj = box
        hops = Tuple{DataType,Vector{Int}}[]
        whole = access.offset < 0
        for offset in access.path
            fields = _deref_fields(typeof(obj), offset)
            if fields === nothing
                whole = true
                break
            end
            push!(hops, (typeof(obj), collect(Int, fields)))
            obj = _getfields(obj, fields)
        end
        leaf = if whole || obj isa AbstractArray
            Leaf(hops, Nothing, 0)
        else
            field = _field_at(typeof(obj), access.offset)
            if field === nothing || !ismutable(obj)
                Leaf(hops, Nothing, 0)
            else
                i = field[1][1]
                # A field holding references only matters through what they
                # point to.
                isbitstype(fieldtype(typeof(obj), i)) || continue
                Leaf(hops, typeof(obj), i)
            end
        end
        any(l -> l.hops == leaf.hops && l.field == leaf.field, leaves) ||
            push!(leaves, leaf)
    end
    return leaves
end

# Fills `sched.live` with the leaves in `box`; false if a type on the way is not
# the one the plan was made for.
function _fill_leaves!(sched, leaves::Vector{Leaf}, box)
    live = sched.live
    empty!(live.objects)
    empty!(live.scalars)
    empty!(sched.owners)
    for leaf in leaves
        obj = box
        for (T, fields) in leaf.hops
            typeof(obj) === T || return false
            for f in fields
                obj = getfield(obj, f)
            end
        end
        if leaf.field == 0
            _push_new!(live.objects, obj)
        else
            typeof(obj) === leaf.owner || return false
            push!(sched.owners, (obj, leaf.field))
            push!(live.scalars, getfield(obj, leaf.field))
        end
    end
    return true
end

function _push_new!(objects::Vector{Any}, obj)
    for o in objects
        o === obj && return objects
    end
    return push!(objects, obj)
end

# The pieces of the loop body in `box` that a snapshot holds, in `sched.live`
# (their owners in `sched.owners`). The plan is made once per schedule, and
# again only if the body holds objects of other types.
function _leaves!(sched, box)
    leaves = sched.leaves
    if leaves === nothing || !_fill_leaves!(sched, leaves, box)
        leaves = sched.leaves = _plan_leaves(sched, box)
        _fill_leaves!(sched, leaves, box)
    end
    return sched.live
end

"""
    EnzymeSchedule

One run of a scheme through the C interface, from `init` to `finalize`.
"""
mutable struct EnzymeSchedule{S,A,ST,I,E}
    # The callbacks that run for each step, compiled for this type of schedule;
    # the scheme table's forward to them, so none dispatches on the schedule.
    # First, at fixed offsets, for `_callback`.
    next_action_cb::Ptr{Cvoid}
    store_cb::Ptr{Cvoid}
    restore_cb::Ptr{Cvoid}
    save_state_cb::Ptr{Cvoid}
    load_state_cb::Ptr{Cvoid}
    scheme::S
    actions::A
    buffer::Vector{UInt8}
    # The scheme's storage, typed: the scheme's own field is abstract.
    storage::ST
    # Periodic: the storage for the steps of the segment being reversed.
    inner::I
    segments::Int
    # With `save_state`, the scheme's storage holds the loop body and the
    # regions go here; without, the scheme's storage holds the regions.
    regions::Union{Nothing,ArrayStorage{Vector{UInt8}}}
    # The driver's own slots: -1, the state the reverse sweep starts from, put
    # back at its end, and -2, the state before the last step.
    entry::E
    entry_regions::ArrayStorage{Vector{UInt8}}
    # What the step accesses through the loop body (from Enzyme's set_paths),
    # and which of it a snapshot holds: nothing = the whole body.
    paths::Union{Nothing,Vector{AccessPath}}
    snapshot::Symbol
    # How to find what a masked snapshot holds (from `paths`), and the snapshot
    # of the live state it is copied from or to, with the objects owning its
    # scalars.
    leaves::Union{Nothing,Vector{Leaf}}
    live::MaskedSnapshot
    owners::Vector{Tuple{Any,Int}}
    # Bytes copied into snapshots, for LAST_SNAPSHOT_BYTES: the size of the
    # first snapshot, and how many were taken.
    snapshot_size::Int
    snapshots::Int
end

"""
    LAST_SNAPSHOT_BYTES[]

The bytes the last schedule run through Enzyme copied into snapshots of the loop
body, summed over all of them (each counted at the size of the first). For
inspecting what `EnzymeLLVM`'s `snapshot` option saves.
"""
const LAST_SNAPSHOT_BYTES = Ref(0)

# A schedule lives from `init` to `finalize`, across calls from C; this roots it.
const LIVE_SCHEDULES = IdDict{EnzymeSchedule,Nothing}()

_actions(scheme::Revolve) = scheme

# The storage and index for the snapshot of the state before `step`, in
# `slot`, to `store` or to restore.
_slot_storage(s::EnzymeSchedule{<:Revolve}, slot, step, store) = (s.storage, slot + 1)

function _slot_storage(s::EnzymeSchedule{<:Periodic}, slot, step, store)
    slot < s.segments && return (s.storage, slot + 1)
    return (s.inner, slot - s.segments + 1)
end

# Online_r2 keeps its snapshots by step, as its own driver does: the offline
# Revolve that takes over once the loop has ended numbers slots differently.
function _slot_storage(s::EnzymeSchedule{<:Online_r2}, slot, step, store)
    a = s.actions::OnlineActions
    if store
        if a.revolve === nothing
            # The online phase overwrites its slot.
            i = slot + 1
            let i = i
                filter!(kv -> kv.second != i, a.storemap)
            end
        else
            i = pop!(a.free)
        end
        a.storemap[step] = i
    else
        i = a.storemap[step]
    end
    return (s.storage, i)
end

# The storage and index for the state (loop body or regions) of `slot`.
_state_storage(s::EnzymeSchedule, slot, step, store) =
    slot < 0 ? (s.entry, -slot) : _slot_storage(s, slot, step, store)

# The storage and index for the regions of `slot`.
function _region_storage(s::EnzymeSchedule, slot, step, store)
    s.regions === nothing && return _state_storage(s, slot, step, store)
    slot < 0 && return (s.entry_regions, -slot)
    return (s.regions, _slot_storage(s, slot, step, store)[2])
end

_slots(scheme::Revolve) = scheme.acp
_slots(scheme::Periodic) = scheme.acp + (scheme.steps == 0 ? 0 : scheme.period)
_slots(scheme::Online_r2) = scheme.acp

function _new_schedule(scheme, actions, inner, segments, bytes, ::Type{FT}) where {FT}
    state = FT !== Vector{UInt8}
    regions = state ? ArrayStorage{Vector{UInt8}}(max(_slots(scheme), 1)) : nothing
    entry = ArrayStorage{FT}(2)
    sched = EnzymeSchedule(
        C_NULL,
        C_NULL,
        C_NULL,
        C_NULL,
        C_NULL,
        scheme,
        actions,
        # Sized on first use: array storage does not need it (see `_store!`).
        UInt8[],
        scheme.storage,
        inner,
        segments,
        regions,
        entry,
        ArrayStorage{Vector{UInt8}}(2),
        nothing,
        :all,
        nothing,
        MaskedSnapshot(Any[], Any[]),
        Tuple{Any,Int}[],
        0,
        0,
    )
    return _set_callbacks!(sched)
end

# `FT` is what a snapshot is: `Vector{UInt8}` for the bytes of the regions, or
# the loop body (or what of it Enzyme says the step accesses).
_for_loop(alg, steps) =
    steps >= 0 || error(
        "Checkpointing.jl: $(nameof(typeof(alg))) needs the number of iterations; use Online_r2 for a while loop",
    )

"""
    OnlineActions

The schedule of [`Online_r2`](@ref) as actions. While the loop runs they are
the online scheme's; once it has ended, those of the offline Revolve that takes
over, as in `rev_checkpoint_while`: that schedule is over one step more than the
loop ran, which its first turn consumes, and its first turn of a real step is
the forward sweep's last action. Snapshots are kept by step.
"""
mutable struct OnlineActions{S,R}
    online::S
    oldcapo::Int
    # The offline Revolve, once the loop has ended.
    revolve::Union{Nothing,R}
    skipped::Bool
    turned::Bool
    # step => storage index, and the free indices once the loop has ended
    storemap::Dict{Int,Int}
    free::Vector{Int}
end

function next_action!(a::OnlineActions)::Action
    if a.revolve === nothing
        next = next_action!(a.online)
        if next.actionflag == store
            step = next.iteration + 1
            return Action(store, step, step, next.cpnum)
        elseif next.actionflag == forward
            action = Action(forward, next.iteration, a.oldcapo, next.cpnum)
            a.oldcapo = next.iteration
            return action
        end
        error(
            "[Checkpointing.jl]: Online_r2 with $(a.online.acp) checkpoints supports loops " *
            "of at most $(online_r2_limit(a.online.acp)) iterations.",
        )
    end
    next = next_action!(a.revolve)
    flag = next.actionflag
    if flag == firstuturn && !a.skipped
        a.skipped = true
        return next_action!(a)
    elseif flag == uturn || flag == firstuturn
        flag = a.turned ? uturn : firstuturn
        a.turned = true
        step = next.iteration - 1
        if haskey(a.storemap, step)
            push!(a.free, a.storemap[step])
            delete!(a.storemap, step)
        end
        return Action(flag, next.iteration, step, next.cpnum)
    elseif flag == store || flag == restore
        return Action(flag, next.iteration, next.iteration, next.cpnum)
    end
    return next
end

function set_nsteps!(a::OnlineActions, steps::Int)
    a.revolve = update_revolve(a.online, steps + 1)
    held = Set(values(a.storemap))
    a.free = [i for i = 1:a.online.acp if !(i in held)]
    return a
end
set_nsteps!(a, steps::Int) = a

function enzyme_schedule(
    alg::Online_r2{Nothing},
    steps::Int,
    bytes::Int,
    ::Type{FT},
) where {FT}
    steps < 0 || error("Checkpointing.jl: Online_r2 schedules while loops")
    scheme = instantiate(FT, alg)
    actions = OnlineActions{typeof(scheme),Revolve{FT,ArrayStorage{FT}}}(
        scheme,
        0,
        nothing,
        false,
        false,
        Dict{Int,Int}(),
        Int[],
    )
    return _new_schedule(scheme, actions, nothing, 0, bytes, FT)
end

function enzyme_schedule(
    alg::Revolve{Nothing},
    steps::Int,
    bytes::Int,
    ::Type{FT},
) where {FT}
    _for_loop(alg, steps)
    scheme = instantiate(FT, alg, steps)
    return _new_schedule(scheme, scheme, nothing, 0, bytes, FT)
end

function enzyme_schedule(
    alg::Periodic{Nothing},
    steps::Int,
    bytes::Int,
    ::Type{FT},
) where {FT}
    _for_loop(alg, steps)
    scheme = instantiate(FT, alg, steps)
    actions = PeriodicActions(steps, scheme.acp)
    period = steps == 0 ? 0 : cld(steps, actions.segments)
    inner = ArrayStorage{FT}(max(period, 1))
    return _new_schedule(scheme, actions, inner, actions.segments, bytes, FT)
end

function enzyme_schedule(alg::Scheme, steps::Int, bytes::Int, ::Type)
    error(
        "Checkpointing.jl: $(typeof(alg)) cannot schedule a checkpointed for loop through Enzyme",
    )
end

function _pack!(buffer::Vector{UInt8}, regions::Ptr{EnzymeCkptRegion}, n)
    off = 0
    for r = 1:n
        region = unsafe_load(regions, r)
        region.addrspace == 0 || error(
            "Checkpointing.jl: snapshots of address space $(region.addrspace) are not supported yet",
        )
        GC.@preserve buffer unsafe_copyto!(
            pointer(buffer, off + 1),
            region.ptr,
            region.bytes,
        )
        off += region.bytes
    end
    return buffer
end

function _unpack!(regions::Ptr{EnzymeCkptRegion}, n, buffer::Vector{UInt8})
    off = 0
    for r = 1:n
        region = unsafe_load(regions, r)
        GC.@preserve buffer unsafe_copyto!(
            region.ptr,
            pointer(buffer, off + 1),
            region.bytes,
        )
        off += region.bytes
    end
    return nothing
end

_schedule(state::Ptr{Cvoid}) = unsafe_pointer_to_objref(state)::EnzymeSchedule

function _enzyme_init(data::Ptr{Cvoid}, nsteps::Int64, bytes::UInt64, state::Bool)
    sched = _schedule_for(unsafe_pointer_to_objref(data), Int(nsteps), Int(bytes), state)
    LIVE_SCHEDULES[sched] = nothing
    return pointer_from_objref(sched)
end

_schedule_for(alg::Scheme, nsteps, bytes, state) =
    enzyme_schedule(alg, nsteps, bytes, state ? Any : Vector{UInt8})

_enzyme_init_regions(data::Ptr{Cvoid}, nsteps::Int64, bytes::UInt64)::Ptr{Cvoid} =
    _enzyme_init(data, nsteps, bytes, false)
_enzyme_init_state(data::Ptr{Cvoid}, nsteps::Int64, bytes::UInt64)::Ptr{Cvoid} =
    _enzyme_init(data, nsteps, bytes, true)

# The callbacks that run for each step call the `k`th callback of the schedule
# at `state`, compiled for its type (`_schedule` cannot know it; see
# `_set_callbacks!`).
_callback(state::Ptr{Cvoid}, k) = unsafe_load(Ptr{Ptr{Cvoid}}(state), k)

function _enzyme_next_action(state::Ptr{Cvoid}, out::Ptr{Action})::Cvoid
    ccall(_callback(state, 1), Cvoid, (Ptr{Cvoid}, Ptr{Action}), state, out)
    return nothing
end

function _next_action!(sched::EnzymeSchedule, out::Ptr{Action})
    unsafe_store!(out, next_action!(sched.actions))
    return nothing
end

function _enzyme_store(
    state::Ptr{Cvoid},
    slot::Int64,
    step::Int64,
    regions::Ptr{EnzymeCkptRegion},
    n::UInt64,
)::Cvoid
    ccall(
        _callback(state, 2),
        Cvoid,
        (Ptr{Cvoid}, Int64, Int64, Ptr{EnzymeCkptRegion}, UInt64),
        state,
        slot,
        step,
        regions,
        n,
    )
    return nothing
end

function _store!(sched::EnzymeSchedule, slot, step, regions, n)
    storage, i = _region_storage(sched, slot, step, true)
    _store_regions!(storage, i, sched.buffer, regions, n)
    return nothing
end

_region_bytes(regions::Ptr{EnzymeCkptRegion}, n) =
    sum(r -> Int(unsafe_load(regions, r).bytes), 1:n; init = 0)

# In memory, the regions are copied into the slot itself, as Enzyme's own
# schemes do; any other storage takes them packed into `buffer`.
function _store_regions!(storage::ArrayStorage{Vector{UInt8}}, i, buffer, regions, n)
    slots = storage._fstorage
    checkbounds(slots, i)
    bytes = _region_bytes(regions, n)
    if !isassigned(slots, i)
        @inbounds slots[i] = Vector{UInt8}(undef, bytes)
    end
    @inbounds slot = slots[i]
    length(slot) == bytes || resize!(slot, bytes)
    _pack!(slot, regions, n)
    return storage
end

function _store_regions!(storage, i, buffer, regions, n)
    resize!(buffer, _region_bytes(regions, n))
    return save!(storage, _pack!(buffer, regions, n), i)
end

function _enzyme_restore(
    state::Ptr{Cvoid},
    slot::Int64,
    step::Int64,
    regions::Ptr{EnzymeCkptRegion},
    n::UInt64,
)::Cvoid
    ccall(
        _callback(state, 3),
        Cvoid,
        (Ptr{Cvoid}, Int64, Int64, Ptr{EnzymeCkptRegion}, UInt64),
        state,
        slot,
        step,
        regions,
        n,
    )
    return nothing
end

function _restore!(sched::EnzymeSchedule, slot, step, regions, n)
    storage, i = _region_storage(sched, slot, step, false)
    _restore_regions!(regions, n, storage, i, sched.buffer)
    return nothing
end

function _restore_regions!(regions, n, storage::ArrayStorage{Vector{UInt8}}, i, buffer)
    slots = storage._fstorage
    checkbounds(slots, i)
    isassigned(slots, i) || throw(
        ArgumentError(
            "[Checkpointing.jl]: checkpoint $i was restored before it was stored.",
        ),
    )
    return _unpack!(regions, n, @inbounds slots[i])
end

function _restore_regions!(regions, n, storage, i, buffer)
    resize!(buffer, _region_bytes(regions, n))
    load!(buffer, storage, i)
    return _unpack!(regions, n, buffer)
end

# The loop body of EnzymeCore.checkpoint_for: its box is the first word of the
# step's environment.
_loop_body(env::Ptr{Cvoid}) = unsafe_pointer_to_objref(unsafe_load(Ptr{Ptr{Cvoid}}(env)))[]

_loop_box(env::Ptr{Cvoid}) = unsafe_pointer_to_objref(unsafe_load(Ptr{Ptr{Cvoid}}(env)))

# A mask has to save at least this fraction of a snapshot of the whole loop
# body to be used: following it costs more per snapshot than copying the body,
# so a mask that saves (next to) nothing only makes snapshots slower.
const MASK_MIN_SAVING = 0.05

# Whether snapshots hold only what the step accesses. Decided at the first
# snapshot, before any is taken, and kept for the schedule: without the paths,
# with `snapshot = :all`, or if the mask saves too little, snapshots are of the
# whole body.
function _masked!(sched, box)
    (sched.paths === nothing || sched.snapshot === :all) && return false
    sched.leaves === nothing || return true
    leaves = _plan_leaves(sched, box)
    _fill_leaves!(sched, leaves, box)
    whole = Base.summarysize(box[])
    if Base.summarysize(sched.live) > (1 - MASK_MIN_SAVING) * whole
        sched.paths = nothing
        return false
    end
    sched.leaves = leaves
    return true
end

function _enzyme_save_state(
    state::Ptr{Cvoid},
    slot::Int64,
    step::Int64,
    env::Ptr{Cvoid},
)::Cvoid
    ccall(
        _callback(state, 4),
        Cvoid,
        (Ptr{Cvoid}, Int64, Int64, Ptr{Cvoid}),
        state,
        slot,
        step,
        env,
    )
    return nothing
end

_save_state_env!(sched::EnzymeSchedule, slot, step, env) =
    _save_state!(sched, slot, step, _loop_box(env))

# What snapshots of the loop body are: its type, or with a mask also
# MaskedSnapshot.
_snapshot_type(::EnzymeSchedule{S,A,ST,I,ArrayStorage{FT}}) where {S,A,ST,I,FT} = FT

function _save_state!(sched::EnzymeSchedule, slot, step, box)
    storage, i = _state_storage(sched, slot, step, true)
    if _masked!(sched, box)
        snap = _leaves!(sched, box)
        _count_snapshot!(sched, snap)
        save!(storage, snap, i)
    else
        body = box[]::_snapshot_type(sched)
        _count_snapshot!(sched, body)
        save!(storage, body, i)
    end
    return nothing
end

# Measuring a snapshot walks all of it, so only the first is.
function _count_snapshot!(sched::EnzymeSchedule, snap)
    sched.snapshots == 0 && (sched.snapshot_size = Base.summarysize(snap))
    sched.snapshots += 1
    return nothing
end

function _enzyme_load_state(
    state::Ptr{Cvoid},
    slot::Int64,
    step::Int64,
    env::Ptr{Cvoid},
)::Cvoid
    ccall(
        _callback(state, 5),
        Cvoid,
        (Ptr{Cvoid}, Int64, Int64, Ptr{Cvoid}),
        state,
        slot,
        step,
        env,
    )
    return nothing
end

_load_state_env!(sched::EnzymeSchedule, slot, step, env) =
    _load_state!(sched, slot, step, _loop_box(env))

# Defined after the callbacks it compiles.
function _set_callbacks!(sched::S) where {S<:EnzymeSchedule}
    for k = 1:5
        fieldoffset(S, k) == (k - 1) * sizeof(Ptr{Cvoid}) || error("unreachable")
    end
    sched.next_action_cb = @cfunction(_next_action!, Cvoid, (Ref{S}, Ptr{Action}))
    sched.store_cb =
        @cfunction(_store!, Cvoid, (Ref{S}, Int64, Int64, Ptr{EnzymeCkptRegion}, UInt64))
    sched.restore_cb =
        @cfunction(_restore!, Cvoid, (Ref{S}, Int64, Int64, Ptr{EnzymeCkptRegion}, UInt64))
    # Snapshots of the regions only need the above; not compiling the loop
    # body's callbacks for them keeps the code juliac --trim has to compile
    # free of the dynamically typed masked snapshots.
    if _snapshot_type(sched) !== Vector{UInt8}
        sched.save_state_cb =
            @cfunction(_save_state_env!, Cvoid, (Ref{S}, Int64, Int64, Ptr{Cvoid}))
        sched.load_state_cb =
            @cfunction(_load_state_env!, Cvoid, (Ref{S}, Int64, Int64, Ptr{Cvoid}))
    end
    return sched
end

function _load_state!(sched::EnzymeSchedule, slot, step, box)
    storage, i = _state_storage(sched, slot, step, false)
    if _masked!(sched, box)
        live = _leaves!(sched, box)
        # Copies the stored objects into the live ones.
        load!(live, storage, i)
        for (k, (o, f)) in enumerate(sched.owners)
            setfield!(o, f, live.scalars[k])
        end
    else
        load!(box[]::_snapshot_type(sched), storage, i)
    end
    return nothing
end

function _enzyme_set_nsteps(state::Ptr{Cvoid}, n::Int64)::Cvoid
    set_nsteps!(_schedule(state).actions, Int(n))
    return nothing
end

function _enzyme_set_paths(state::Ptr{Cvoid}, paths::Ptr{Int64}, len::UInt64)::Cvoid
    sched = _schedule(state)
    sched.paths = decode_paths(paths, len)
    sched.leaves = nothing
    return nothing
end

function _enzyme_finalize(state::Ptr{Cvoid})::Cvoid
    sched = _schedule(state)
    LAST_SNAPSHOT_BYTES[] = sched.snapshot_size * sched.snapshots
    delete!(LIVE_SCHEDULES, sched)
    return nothing
end

const ENZYME_VTABLE = Ref{EnzymeCheckpointScheme}()
const ENZYME_VTABLE_STATE = Ref{EnzymeCheckpointScheme}()

# The tables are filled in on first use rather than in `__init__`, so a program
# that never hands a scheme to Enzyme through them -- like a library built with
# juliac that has its own -- does not compile them.
const _abi_initialized = Ref(false)

function _init_enzyme_abi()
    next = @cfunction(_enzyme_next_action, Cvoid, (Ptr{Cvoid}, Ptr{Action}))
    store = @cfunction(
        _enzyme_store,
        Cvoid,
        (Ptr{Cvoid}, Int64, Int64, Ptr{EnzymeCkptRegion}, UInt64)
    )
    restore = @cfunction(
        _enzyme_restore,
        Cvoid,
        (Ptr{Cvoid}, Int64, Int64, Ptr{EnzymeCkptRegion}, UInt64)
    )
    finalize = @cfunction(_enzyme_finalize, Cvoid, (Ptr{Cvoid},))
    set_nsteps = @cfunction(_enzyme_set_nsteps, Cvoid, (Ptr{Cvoid}, Int64))
    ENZYME_VTABLE[] = EnzymeCheckpointScheme(
        ENZYME_CKPT_ABI_VERSION,
        @cfunction(_enzyme_init_regions, Ptr{Cvoid}, (Ptr{Cvoid}, Int64, UInt64)),
        next,
        store,
        restore,
        set_nsteps,
        finalize,
        C_NULL,
        C_NULL,
        C_NULL,
    )
    ENZYME_VTABLE_STATE[] = EnzymeCheckpointScheme(
        ENZYME_CKPT_ABI_VERSION,
        @cfunction(_enzyme_init_state, Ptr{Cvoid}, (Ptr{Cvoid}, Int64, UInt64)),
        next,
        store,
        restore,
        set_nsteps,
        finalize,
        @cfunction(_enzyme_save_state, Cvoid, (Ptr{Cvoid}, Int64, Int64, Ptr{Cvoid})),
        @cfunction(_enzyme_load_state, Cvoid, (Ptr{Cvoid}, Int64, Int64, Ptr{Cvoid})),
        @cfunction(_enzyme_set_paths, Cvoid, (Ptr{Cvoid}, Ptr{Int64}, UInt64)),
    )
    _abi_initialized[] = true
    return nothing
end

"""
    enzyme_scheme(alg::Scheme; state = false) -> (scheme::Ptr{Cvoid}, data::Ptr{Cvoid})

The two pointers that make `alg` the scheme of a loop Enzyme differentiates:
pass them as `enzyme_scheme, scheme, data` to `__enzyme_checkpoint_for`, or
through the code that does. `alg` is a scheme as given to `@ad_checkpoint`, for
example `Revolve(3)` or `Periodic(4; storage = HDF5Storage)`; each run of the
loop instantiates it anew. Keep `alg` alive (`GC.@preserve`) until the reverse
pass has run.

Snapshots are of the memory regions Enzyme hands over. With `state = true` they
are copies of the loop body of `EnzymeCore.checkpoint_for` instead (see
[`EnzymeLLVM`](@ref)).
"""
function enzyme_scheme(alg::Scheme; state::Bool = false)
    _abi_initialized[] || _init_enzyme_abi()
    vtable = state ? ENZYME_VTABLE_STATE : ENZYME_VTABLE
    return (
        Ptr{Cvoid}(Base.unsafe_convert(Ptr{EnzymeCheckpointScheme}, vtable)),
        pointer_from_objref(alg),
    )
end

"""
    EnzymeLLVM(alg::Scheme; snapshot = :accessed)

`alg`, applied by Enzyme's LLVM core rather than by this package's EnzymeRules:

    @ad_checkpoint EnzymeLLVM(Revolve(3)) for i = 1:n
        ...
    end

Enzyme differentiates each iteration once, at compile time, and runs the
schedule of `alg` itself; snapshots are kept in the storage of `alg`. For loops
take Revolve or Periodic, while loops Online_r2.

Enzyme tells the scheme what the loop body accesses, and `snapshot` says which
of it a snapshot holds:

- `:accessed`: what an iteration reads or writes, down to arrays and the scalar
  fields of mutable structs. State the loop never touches is not copied.
- `:written`: only what an iteration writes. Read-only state (parameters) is
  not copied either, which is only correct if nothing changes it between the
  loop and the end of the reverse pass.
- `:all`: the whole loop body, as the EnzymeRules path does.

With `:accessed` or `:written`, a snapshot is of the whole body after all if
leaving out what the step does not need saves less than 5% of it: copying just
those pieces costs more per snapshot than copying the body.
"""
mutable struct EnzymeLLVM{S<:Scheme}
    scheme::S
    snapshot::Symbol
    function EnzymeLLVM(scheme::S; snapshot::Symbol = :accessed) where {S<:Scheme}
        snapshot in (:accessed, :written, :all) ||
            throw(ArgumentError("snapshot must be :accessed, :written or :all"))
        return new{S}(scheme, snapshot)
    end
end

# The data pointer of one loop through EnzymeLLVM: its scheme, and the type of
# its body, which snapshots are copies of.
mutable struct EnzymeLLVMRun{B,E<:EnzymeLLVM}
    alg::E
end

EnzymeLLVMRun(alg::E, body::B) where {B,E<:EnzymeLLVM} = EnzymeLLVMRun{B,E}(alg)

function _schedule_for(run::EnzymeLLVMRun{B}, nsteps, bytes, state) where {B}
    alg = run.alg
    # Snapshots of the loop body keep its type, so that copying them is
    # compiled for it; with a mask they may also be of what the step accesses.
    FT = !state ? Vector{UInt8} : alg.snapshot === :all ? B : Union{B,MaskedSnapshot}
    sched = enzyme_schedule(alg.scheme, nsteps, bytes, FT)
    sched.snapshot = alg.snapshot
    return sched
end

function checkpoint_while(body::Function, alg::EnzymeLLVM)
    scheme = enzyme_scheme(alg.scheme; state = true)[1]
    run = EnzymeLLVMRun(alg, body)
    data = pointer_from_objref(run)
    GC.@preserve run EnzymeCore.checkpoint_while(scheme, data, body)
    return nothing
end

function checkpoint_for(body::Function, alg::EnzymeLLVM, range::UnitRange{Int})
    scheme = enzyme_scheme(alg.scheme; state = true)[1]
    run = EnzymeLLVMRun(alg, body)
    data = pointer_from_objref(run)
    GC.@preserve run EnzymeCore.checkpoint_for(
        scheme,
        data,
        first(range),
        length(range),
        body,
    )
    return nothing
end
