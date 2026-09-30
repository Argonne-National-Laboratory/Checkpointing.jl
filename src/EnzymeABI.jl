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
end

# A schedule lives from `init` to `finalize`, across calls from C; this roots it.
const LIVE_SCHEDULES = IdDict{EnzymeSchedule,Nothing}()

_actions(scheme::Revolve) = scheme
_slot_storage(s::EnzymeSchedule{<:Revolve}, slot) = (s.scheme.storage, slot + 1)

function _slot_storage(s::EnzymeSchedule{<:Periodic}, slot)
    slot < s.segments && return (s.scheme.storage, slot + 1)
    return (s.inner, slot - s.segments + 1)
end

function enzyme_schedule(alg::Revolve{Nothing}, steps::Int, bytes::Int)
    scheme = instantiate(Vector{UInt8}, alg, steps)
    return EnzymeSchedule(scheme, _actions(scheme), zeros(UInt8, bytes), nothing, 0)
end

function enzyme_schedule(alg::Periodic{Nothing}, steps::Int, bytes::Int)
    scheme = instantiate(Vector{UInt8}, alg, steps)
    actions = PeriodicActions(steps, scheme.acp)
    period = steps == 0 ? 0 : cld(steps, actions.segments)
    inner = ArrayStorage{Vector{UInt8}}(max(period, 1))
    return EnzymeSchedule(scheme, actions, zeros(UInt8, bytes), inner, actions.segments)
end

function enzyme_schedule(alg::Scheme, steps::Int, bytes::Int)
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

function _enzyme_init(data::Ptr{Cvoid}, nsteps::Int64, bytes::UInt64)::Ptr{Cvoid}
    alg = unsafe_pointer_to_objref(data)::Scheme
    nsteps < 0 &&
        error("Checkpointing.jl: while loops are not supported through Enzyme yet")
    sched = enzyme_schedule(alg, Int(nsteps), Int(bytes))
    LIVE_SCHEDULES[sched] = nothing
    return pointer_from_objref(sched)
end

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
    storage, i = _slot_storage(sched, slot)
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
    storage, i = _slot_storage(sched, slot)
    load!(sched.buffer, storage, i)
    _unpack!(regions, n, sched.buffer)
    return nothing
end

function _enzyme_finalize(state::Ptr{Cvoid})::Cvoid
    delete!(LIVE_SCHEDULES, _schedule(state))
    return nothing
end

const ENZYME_VTABLE = Ref{EnzymeCheckpointScheme}()

function _init_enzyme_abi()
    ENZYME_VTABLE[] = EnzymeCheckpointScheme(
        ENZYME_CKPT_ABI_VERSION,
        @cfunction(_enzyme_init, Ptr{Cvoid}, (Ptr{Cvoid}, Int64, UInt64)),
        @cfunction(_enzyme_next_action, Cvoid, (Ptr{Cvoid}, Ptr{Action})),
        @cfunction(
            _enzyme_store,
            Cvoid,
            (Ptr{Cvoid}, Int64, Int64, Ptr{EnzymeCkptRegion}, UInt64)
        ),
        @cfunction(
            _enzyme_restore,
            Cvoid,
            (Ptr{Cvoid}, Int64, Int64, Ptr{EnzymeCkptRegion}, UInt64)
        ),
        C_NULL,
        @cfunction(_enzyme_finalize, Cvoid, (Ptr{Cvoid},)),
        C_NULL,
        C_NULL,
    )
    return nothing
end

"""
    enzyme_scheme(alg::Scheme) -> (scheme::Ptr{Cvoid}, data::Ptr{Cvoid})

The two pointers that make `alg` the scheme of a loop Enzyme differentiates:
pass them as `enzyme_scheme, scheme, data` to `__enzyme_checkpoint_for`, or
through the code that does. `alg` is a scheme as given to `@ad_checkpoint`, for
example `Revolve(3)` or `Periodic(4; storage = HDF5Storage)`; each run of the
loop instantiates it anew. Keep `alg` alive (`GC.@preserve`) until the reverse
pass has run.
"""
function enzyme_scheme(alg::Scheme)
    return (
        Ptr{Cvoid}(Base.unsafe_convert(Ptr{EnzymeCheckpointScheme}, ENZYME_VTABLE)),
        pointer_from_objref(alg),
    )
end
