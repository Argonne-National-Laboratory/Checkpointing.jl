# Drives Checkpointing.jl's EnzymeLLVM scheme table for a concrete loop body as
# Enzyme's driver does (init, set_paths, next_action, save_state/load_state of
# the boxed body, finalize), without AD, so it can be compiled with
# juliac --trim=safe --output-exe. Checks that every restore gives back the
# state stored in that slot, for :all, :accessed and :written snapshots.
module HeatDrive
using Checkpointing
const C = Checkpointing

mutable struct Heat
    T::Vector{Float64}
    Tnext::Vector{Float64}
    κ::Vector{Float64}
    diag::Vector{Float64}
    t::Float64
end

function heat_step!(h::Heat)
    T, Tn, κ = h.T, h.Tnext, h.κ
    @inbounds for k = 2:(length(T)-1)
        Tn[k] = T[k] + κ[k] * (T[k+1] - 2T[k] + T[k-1]) + 0.01 * sin(T[k] * h.t)
    end
    @inbounds for k = 2:(length(T)-1)
        T[k] = Tn[k]
    end
    h.t += 0.1
    return nothing
end

newheat() = Heat([sin(k / 10) for k = 1:100], zeros(100), fill(0.2, 100), ones(1000), 0.5)

# The accesses Enzyme reports for heat_step! (set_paths): count, offsets of
# the pointers followed, offset of the access (-1 whole), flags (1 r, 2 w).
const PATHS = Int64[
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    8,
    1,
    1,
    0,
    16,
    1,
    1,
    0,
    32,
    3,
    2,
    0,
    0,
    0,
    1,
    2,
    0,
    0,
    8,
    1,
    2,
    0,
    0,
    16,
    1,
    3,
    0,
    0,
    0,
    -1,
    3,
    2,
    0,
    8,
    0,
    1,
    2,
    0,
    8,
    8,
    1,
    3,
    0,
    8,
    0,
    -1,
    3,
    2,
    0,
    16,
    0,
    1,
    2,
    0,
    16,
    8,
    1,
    3,
    0,
    16,
    0,
    -1,
    1,
]

same(a::Heat, b::Heat) = a.T == b.T && a.Tnext == b.Tnext && a.κ == b.κ && a.t == b.t

function drive(snapshot::Symbol, nsteps::Int, nsnap::Int)
    h = newheat()
    body = let h = h
        i -> heat_step!(h)
    end
    run = C.EnzymeLLVMRun(EnzymeLLVM(Revolve(nsnap); snapshot = snapshot), body)
    table = unsafe_load(Ptr{C.EnzymeCheckpointScheme}(C.enzyme_scheme(run)))
    box = Ref(body)
    env = Ref(Ptr{Cvoid}(pointer_from_objref(box)))
    data = pointer_from_objref(run)
    fails = 0
    GC.@preserve run box env begin
        envp = Ptr{Cvoid}(Base.unsafe_convert(Ptr{Ptr{Cvoid}}, env))
        state = ccall(table.init, Ptr{Cvoid}, (Ptr{Cvoid}, Int64, UInt64), data, nsteps, 0)
        ccall(
            table.set_paths,
            Cvoid,
            (Ptr{Cvoid}, Ptr{Int64}, UInt64),
            state,
            PATHS,
            length(PATHS),
        )
        # The state before each step, and what each slot holds.
        ref = Heat[]
        held = Dict{Int,Heat}()
        snap(x::Heat) = Heat(copy(x.T), copy(x.Tnext), copy(x.κ), copy(x.diag), x.t)
        # The driver's slot -1: the state the reverse pass starts from.
        ccall(
            table.save_state,
            Cvoid,
            (Ptr{Cvoid}, Int64, Int64, Ptr{Cvoid}),
            state,
            -1,
            0,
            envp,
        )
        held[-1] = snap(h)
        action = Ref(C.Action(C.none, 0, 0, 0))
        while true
            ccall(table.next_action, Cvoid, (Ptr{Cvoid}, Ptr{C.Action}), state, action)
            a = action[]
            if a.actionflag == C.store
                ccall(
                    table.save_state,
                    Cvoid,
                    (Ptr{Cvoid}, Int64, Int64, Ptr{Cvoid}),
                    state,
                    a.cpnum,
                    a.startiteration,
                    envp,
                )
                held[a.cpnum] = snap(h)
            elseif a.actionflag == C.restore
                h.T .= -1.0
                h.Tnext .= -1.0
                h.t = -1.0  # clobber what a step changes
                ccall(
                    table.load_state,
                    Cvoid,
                    (Ptr{Cvoid}, Int64, Int64, Ptr{Cvoid}),
                    state,
                    a.cpnum,
                    a.startiteration,
                    envp,
                )
                same(h, held[a.cpnum]) || (fails += 1)
            elseif a.actionflag == C.forward
                for i = a.startiteration:(a.iteration-1)
                    body(i)
                end
            elseif a.actionflag == C.firstuturn || a.actionflag == C.uturn
                body(a.startiteration)
            elseif a.actionflag == C.done
                break
            else
                fails += 1
                break
            end
        end
        h.T .= 0.0
        h.t = 0.0
        ccall(
            table.load_state,
            Cvoid,
            (Ptr{Cvoid}, Int64, Int64, Ptr{Cvoid}),
            state,
            -1,
            0,
            envp,
        )
        same(h, held[-1]) || (fails += 1)
        ccall(table.finalize, Cvoid, (Ptr{Cvoid},), state)
    end
    # A plain run of the same steps ends where the restored start state does
    # after them.
    p = newheat()
    for i = 1:nsteps
        heat_step!(p)
    end
    for i = 1:nsteps
        heat_step!(h)
    end
    same(h, p) || (fails += 1)
    return fails, C.LAST_SNAPSHOT_BYTES[]
end

function run_all()
    total = 0
    for snapshot in Symbol[:all, :accessed, :written],
        (n, c) in Tuple{Int,Int}[(30, 4), (10, 2), (7, 7), (1, 1)]

        fails, bytes = drive(snapshot, n, c)
        println(
            Core.stdout,
            "snapshot=" *
            String(snapshot) *
            " steps=" *
            string(n) *
            " snaps=" *
            string(c) *
            " fails=" *
            string(fails) *
            " snapshot bytes=" *
            string(bytes),
        )
        total += fails
    end
    println(Core.stdout, total == 0 ? "ok" : "FAILED")
    return total == 0 ? 0 : 1
end
end

function (@main)(ARGS)
    return HeatDrive.run_all()
end
