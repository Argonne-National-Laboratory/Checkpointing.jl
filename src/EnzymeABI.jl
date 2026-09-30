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

function _queue_segment!(p::PeriodicActions, k, first::Bool)
    s, e, K = _segstart(p, k), _segstart(p, k + 1), p.segments
    for j = s:(e-2)
        push!(p.queue, Action(store, j, j, K + (j - s)))
        push!(p.queue, Action(forward, j + 1, j, K + (j - s)))
    end
    push!(p.queue, Action(first ? firstuturn : uturn, e, e - 1, K + (e - 1 - s)))
    for j = (e-2):-1:s
        push!(p.queue, Action(restore, j, j, K + (j - s)))
        push!(p.queue, Action(uturn, j + 1, j, K + (j - s)))
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
    EnzymeSchedule

One run of a scheme through the C interface, from `init` to `finalize`.
"""
mutable struct EnzymeSchedule{S,A}
    scheme::S
    actions::A
    buffer::Vector{UInt8}
    # Periodic: the storage for the steps of the segment being reversed.
    inner::Union{Nothing,AbstractStorage}
    segments::Int
    # With `save_state`, the scheme's storage holds the loop body and the
    # regions go here; without, the scheme's storage holds the regions.
    regions::Union{Nothing,ArrayStorage{Vector{UInt8}}}
    # The driver's own slots: -1, the state the reverse sweep starts from, put
    # back at its end, and -2, the state before the last step.
    entry::AbstractStorage
    entry_regions::ArrayStorage{Vector{UInt8}}
end

# A schedule lives from `init` to `finalize`, across calls from C; this roots it.
const LIVE_SCHEDULES = IdDict{EnzymeSchedule,Nothing}()

_actions(scheme::Revolve) = scheme
_slot_storage(s::EnzymeSchedule{<:Revolve}, slot) = (s.scheme.storage, slot + 1)

function _slot_storage(s::EnzymeSchedule{<:Periodic}, slot)
    slot < s.segments && return (s.scheme.storage, slot + 1)
    return (s.inner, slot - s.segments + 1)
end

# The storage and index for the state (loop body or regions) of `slot`.
_state_storage(s::EnzymeSchedule, slot) =
    slot < 0 ? (s.entry, -slot) : _slot_storage(s, slot)

# The storage and index for the regions of `slot`.
function _region_storage(s::EnzymeSchedule, slot)
    s.regions === nothing && return _state_storage(s, slot)
    slot < 0 && return (s.entry_regions, -slot)
    return (s.regions, slot + 1)
end

_slots(scheme::Revolve) = scheme.acp
_slots(scheme::Periodic) = scheme.acp + (scheme.steps == 0 ? 0 : scheme.period)

function _new_schedule(scheme, actions, inner, segments, bytes, state::Bool)
    regions = state ? ArrayStorage{Vector{UInt8}}(max(_slots(scheme), 1)) : nothing
    entry = ArrayStorage{state ? Any : Vector{UInt8}}(2)
    return EnzymeSchedule(
        scheme,
        actions,
        zeros(UInt8, bytes),
        inner,
        segments,
        regions,
        entry,
        ArrayStorage{Vector{UInt8}}(2),
    )
end

# With `state`, snapshots are of the loop body (whose type is not known here),
# otherwise of the bytes of the regions.
function enzyme_schedule(alg::Revolve{Nothing}, steps::Int, bytes::Int, state::Bool)
    scheme = instantiate(state ? Any : Vector{UInt8}, alg, steps)
    return _new_schedule(scheme, scheme, nothing, 0, bytes, state)
end

function enzyme_schedule(alg::Periodic{Nothing}, steps::Int, bytes::Int, state::Bool)
    FT = state ? Any : Vector{UInt8}
    scheme = instantiate(FT, alg, steps)
    actions = PeriodicActions(steps, scheme.acp)
    period = steps == 0 ? 0 : cld(steps, actions.segments)
    inner = ArrayStorage{FT}(max(period, 1))
    return _new_schedule(scheme, actions, inner, actions.segments, bytes, state)
end

function enzyme_schedule(alg::Scheme, steps::Int, bytes::Int, state::Bool)
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
    alg = unsafe_pointer_to_objref(data)::Scheme
    nsteps < 0 &&
        error("Checkpointing.jl: while loops are not supported through Enzyme yet")
    sched = enzyme_schedule(alg, Int(nsteps), Int(bytes), state)
    LIVE_SCHEDULES[sched] = nothing
    return pointer_from_objref(sched)
end

_enzyme_init_regions(data::Ptr{Cvoid}, nsteps::Int64, bytes::UInt64)::Ptr{Cvoid} =
    _enzyme_init(data, nsteps, bytes, false)
_enzyme_init_state(data::Ptr{Cvoid}, nsteps::Int64, bytes::UInt64)::Ptr{Cvoid} =
    _enzyme_init(data, nsteps, bytes, true)

function _enzyme_next_action(state::Ptr{Cvoid}, out::Ptr{Action})::Cvoid
    unsafe_store!(out, next_action!(_schedule(state).actions))
    return nothing
end

function _enzyme_store(
    state::Ptr{Cvoid},
    slot::Int64,
    step::Int64,
    regions::Ptr{EnzymeCkptRegion},
    n::UInt64,
)::Cvoid
    sched = _schedule(state)
    storage, i = _region_storage(sched, slot)
    save!(storage, _pack!(sched.buffer, regions, n), i)
    return nothing
end

function _enzyme_restore(
    state::Ptr{Cvoid},
    slot::Int64,
    step::Int64,
    regions::Ptr{EnzymeCkptRegion},
    n::UInt64,
)::Cvoid
    sched = _schedule(state)
    storage, i = _region_storage(sched, slot)
    load!(sched.buffer, storage, i)
    _unpack!(regions, n, sched.buffer)
    return nothing
end

# The loop body of EnzymeCore.checkpoint_for: its box is the first word of the
# step's environment.
_loop_body(env::Ptr{Cvoid}) = unsafe_pointer_to_objref(unsafe_load(Ptr{Ptr{Cvoid}}(env)))[]

function _enzyme_save_state(
    state::Ptr{Cvoid},
    slot::Int64,
    step::Int64,
    env::Ptr{Cvoid},
)::Cvoid
    sched = _schedule(state)
    storage, i = _state_storage(sched, slot)
    save!(storage, _loop_body(env), i)
    return nothing
end

function _enzyme_load_state(
    state::Ptr{Cvoid},
    slot::Int64,
    step::Int64,
    env::Ptr{Cvoid},
)::Cvoid
    sched = _schedule(state)
    storage, i = _state_storage(sched, slot)
    load!(_loop_body(env), storage, i)
    return nothing
end

function _enzyme_finalize(state::Ptr{Cvoid})::Cvoid
    delete!(LIVE_SCHEDULES, _schedule(state))
    return nothing
end

const ENZYME_VTABLE = Ref{EnzymeCheckpointScheme}()
const ENZYME_VTABLE_STATE = Ref{EnzymeCheckpointScheme}()

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
    ENZYME_VTABLE[] = EnzymeCheckpointScheme(
        ENZYME_CKPT_ABI_VERSION,
        @cfunction(_enzyme_init_regions, Ptr{Cvoid}, (Ptr{Cvoid}, Int64, UInt64)),
        next,
        store,
        restore,
        C_NULL,
        finalize,
        C_NULL,
        C_NULL,
    )
    ENZYME_VTABLE_STATE[] = EnzymeCheckpointScheme(
        ENZYME_CKPT_ABI_VERSION,
        @cfunction(_enzyme_init_state, Ptr{Cvoid}, (Ptr{Cvoid}, Int64, UInt64)),
        next,
        store,
        restore,
        C_NULL,
        finalize,
        @cfunction(_enzyme_save_state, Cvoid, (Ptr{Cvoid}, Int64, Int64, Ptr{Cvoid})),
        @cfunction(_enzyme_load_state, Cvoid, (Ptr{Cvoid}, Int64, Int64, Ptr{Cvoid})),
    )
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
    vtable = state ? ENZYME_VTABLE_STATE : ENZYME_VTABLE
    return (
        Ptr{Cvoid}(Base.unsafe_convert(Ptr{EnzymeCheckpointScheme}, vtable)),
        pointer_from_objref(alg),
    )
end

"""
    EnzymeLLVM(alg::Scheme)

`alg`, applied by Enzyme's LLVM core rather than by this package's EnzymeRules:

    @ad_checkpoint EnzymeLLVM(Revolve(3)) for i = 1:n
        ...
    end

Enzyme differentiates each iteration once, at compile time, and runs the
schedule of `alg` itself; snapshots are copies of the loop body, stored in the
storage of `alg`. Only for loops are supported.
"""
struct EnzymeLLVM{S<:Scheme}
    scheme::S
end

function checkpoint_for(body::Function, alg::EnzymeLLVM, range::UnitRange{Int})
    scheme, data = enzyme_scheme(alg.scheme; state = true)
    GC.@preserve alg EnzymeCore.checkpoint_for(
        scheme,
        data,
        first(range),
        length(range),
        body,
    )
    return nothing
end
