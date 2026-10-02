# Checkpointing.jl's schedules behind Enzyme's C checkpoint scheme ABI
# (enzyme/checkpoint.h), compiled with `juliac --trim` into a shared library.
#
# For each scheme the library exports
#
#     const EnzymeCheckpointScheme *enzyme_ckpt_jl_<name>_scheme(void);
#     void *enzyme_ckpt_jl_<name>(int64_t snapshots);
#
# The first is the scheme table, the second the `data` pointer that goes with
# it; pass both as `enzyme_scheme, scheme, data` to __enzyme_checkpoint_for.
# `enzyme_ckpt_jl_free(data)` releases the scheme. Snapshots are of the memory
# regions Enzyme hands over (the scheme has no save_state/load_state), kept in
# Checkpointing.jl's ArrayStorage.
#
# Checkpointing.jl's own table reaches the scheme through an abstractly typed
# object, which `--trim` cannot compile; here each scheme has its own table, so
# init, set_nsteps and finalize know the concrete type. The callbacks that run
# for each step are Checkpointing.jl's, which call the ones compiled for the
# concrete schedule.
module CkptJuliac

using Checkpointing
using Checkpointing:
    EnzymeCheckpointScheme, EnzymeCkptRegion, Action, ENZYME_CKPT_ABI_VERSION

const C = Checkpointing

# What a schedule's snapshot is: the bytes of the regions.
const FT = Vector{UInt8}

# The schemes C holds a `data` pointer to, and the schedules running.
const LIVE = IdDict{Any,Nothing}()

# The schemes as their constructors with the default storage make them.
const NoStorage = Checkpointing.ArrayStorage{Nothing}

# How to make each, as `Revolve(snapshots)` and friends do, but with the
# storage given as an object: given as a type, it is made at run time.
_new_revolve(n) = Revolve{Nothing}(0, n; storage = NoStorage(n))
_new_periodic(n) = Periodic{Nothing}(0, n; storage = NoStorage(n))
_new_online_r2(n) = Online_r2{Nothing}(n; storage = NoStorage(n))

for (name, S) in (
    (:revolve, :(Revolve{Nothing,NoStorage})),
    (:periodic, :(Periodic{Nothing,NoStorage})),
    (:online_r2, :(Online_r2{Nothing,NoStorage})),
)
    Sched = Symbol(:Sched_, name)
    init = Symbol(:_init_, name)
    set_nsteps = Symbol(:_set_nsteps_, name)
    finalize = Symbol(:_finalize_, name)
    table = Symbol(:TABLE_, name)
    getscheme = Symbol(:enzyme_ckpt_jl_, name, :_scheme)
    new = Symbol(:enzyme_ckpt_jl_, name)
    ctor = Symbol(:_new_, name)
    @eval begin
        # The concrete type of a running schedule of this scheme.
        const $Sched =
            Core.Compiler.return_type(C.enzyme_schedule, Tuple{$S,Int,Int,Type{FT}})

        function $init(data::Ptr{Cvoid}, nsteps::Int64, bytes::UInt64)::Ptr{Cvoid}
            alg = unsafe_pointer_to_objref(data)::$S
            sched = C.enzyme_schedule(alg, Int(nsteps), Int(bytes), FT)::$Sched
            LIVE[sched] = nothing
            return pointer_from_objref(sched)
        end

        function $set_nsteps(state::Ptr{Cvoid}, n::Int64)::Cvoid
            sched = unsafe_pointer_to_objref(state)::$Sched
            C.set_nsteps!(sched.actions, Int(n))
            return nothing
        end

        function $finalize(state::Ptr{Cvoid})::Cvoid
            sched = unsafe_pointer_to_objref(state)::$Sched
            delete!(LIVE, sched)
            return nothing
        end

        const $table = Ref{EnzymeCheckpointScheme}()

        Base.@ccallable function $getscheme()::Ptr{Cvoid}
            $table[] = EnzymeCheckpointScheme(
                ENZYME_CKPT_ABI_VERSION,
                @cfunction($init, Ptr{Cvoid}, (Ptr{Cvoid}, Int64, UInt64)),
                @cfunction(C._enzyme_next_action, Cvoid, (Ptr{Cvoid}, Ptr{Action})),
                @cfunction(
                    C._enzyme_store,
                    Cvoid,
                    (Ptr{Cvoid}, Int64, Int64, Ptr{EnzymeCkptRegion}, UInt64)
                ),
                @cfunction(
                    C._enzyme_restore,
                    Cvoid,
                    (Ptr{Cvoid}, Int64, Int64, Ptr{EnzymeCkptRegion}, UInt64)
                ),
                @cfunction($set_nsteps, Cvoid, (Ptr{Cvoid}, Int64)),
                @cfunction($finalize, Cvoid, (Ptr{Cvoid},)),
                C_NULL,
                C_NULL,
                C_NULL,
            )
            return Ptr{Cvoid}(Base.unsafe_convert(Ptr{EnzymeCheckpointScheme}, $table))
        end

        Base.@ccallable function $new(snapshots::Int64)::Ptr{Cvoid}
            alg = $ctor(Int(snapshots))::$S
            LIVE[alg] = nothing
            return pointer_from_objref(alg)
        end
    end
end

Base.@ccallable function enzyme_ckpt_jl_free(data::Ptr{Cvoid})::Cvoid
    delete!(LIVE, unsafe_pointer_to_objref(data))
    return nothing
end

end
