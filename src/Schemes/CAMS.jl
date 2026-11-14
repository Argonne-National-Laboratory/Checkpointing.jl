# Checkpointing for adjoint multistage schemes (CAMS), after Zhang and
# Constantinescu. Revolve only ever stores the state a step starts from, so
# every step is run forward once more when it is adjoined. CAMS may also keep
# what the adjoint of a step actually needs -- in PETSc the stage values of the
# time integrator, here the tape of Enzyme's split reverse mode -- and adjoin
# that step without running it again. A tape is worth more than a state but
# also costs more memory, so the schedule decides per checkpoint which of the
# two to keep.

# --- the schedule ---------------------------------------------------------
#
# Memory is counted in units of one state checkpoint; a tape costs `τ` units.
# Two tables give the least number of forward steps beyond the first sweep
# needed to adjoin `j` steps with `i` units:
#
#   P[i, j]    the state the `j` steps start from is stored, and its
#              checkpoint is one of the `i` units.
#   R[i+1, j+1]  that state is not stored (yet); all `i` units are free.
#
# With the base stored there are three ways to split the `j` steps:
#
#   none        sweep to the end, adjoin the last step, start over:
#                   (j-1) + P[i, j-1]
#   state at k  store the state after `k` steps, adjoin the `j-k` steps behind
#               it with the other `i-1` units, then redo the first `k`:
#                   k + P[i, k] + P[i-1, j-k]
#   tape at k   keep the tape of step `k`, adjoin the `j-k` steps behind it
#               with what is left, adjoin step `k` from its tape, then redo
#               the first `k-1`:
#                   (k-1) + P[i, k-1] + R[i-1-τ, j-k]
#
# Without a stored base the choice is between storing it, P[i, j], and taping
# the first step, R[i-τ, j-1]; a run of the latter is a stretch of the loop
# that is taped outright. These are the recurrences of Lemmas 4.3 and 4.4 of
# the paper with the base checkpoint made explicit.

const _CAMS_INF = typemax(Int) >> 2

# How a segment with a stored base is split: `kind` is one of the three below
# and `k` is where.
const _CAMS_NONE = Int8(0)
const _CAMS_STATE = Int8(1)
const _CAMS_TAPE = Int8(2)

struct CAMSTable
    steps::Int
    units::Int
    stage_units::Int
    P::Matrix{Int}
    R::Matrix{Int}
    kind::Matrix{Int8}
    k::Matrix{Int32}
    exact::Bool
end

# Above this many candidate splits the exhaustive search over `k` is replaced
# by a windowed one. The result is then a valid schedule that is, in practice,
# within a fraction of a percent of the optimal one.
const _CAMS_EXACT_WORK = 4_000_000_000
const _CAMS_WINDOW = 16

# First `k` in `lo:hi` at which `f` stops decreasing. `f` is not convex in
# general, so this only seeds the windowed search.
function _first_rise(f::F, lo::Int, hi::Int) where {F}
    while lo < hi
        mid = (lo + hi) >>> 1
        if f(mid + 1) >= f(mid)
            hi = mid
        else
            lo = mid + 1
        end
    end
    return lo
end

function _scan(f::F, lo::Int, hi::Int, best::Int, bestk::Int) where {F}
    for k = lo:hi
        v = f(k)
        if v < best
            best = v
            bestk = k
        end
    end
    return best, bestk
end

# The least `f(k)` over `1:hi`, exhaustively or in windows around the first
# rise of `f` and around `hint`, the split chosen for the segment one step
# shorter.
function _best_split(f::F, hi::Int, exact::Bool, hint::Int) where {F}
    exact && return _scan(f, 1, hi, _CAMS_INF, 0)
    w = _CAMS_WINDOW
    c = _first_rise(f, 1, hi)
    best, bestk = _scan(f, max(1, c - w), min(hi, c + w), _CAMS_INF, 0)
    h = clamp(hint, 1, hi)
    return _scan(f, max(1, h - w), min(hi, h + w), best, bestk)
end

function CAMSTable(
    steps::Int,
    units::Int,
    stage_units::Int;
    exact::Union{Bool,Nothing} = nothing,
)
    steps >= 0 || throw(ArgumentError("[Checkpointing.jl]: CAMS needs steps >= 0."))
    units >= 1 || throw(ArgumentError("[Checkpointing.jl]: CAMS needs at least one unit."))
    stage_units >= 1 ||
        throw(ArgumentError("[Checkpointing.jl]: CAMS needs stage_units >= 1."))
    m, τ = steps, stage_units
    # With a tape for every step but the last nothing is recomputed, so more
    # units than that change nothing.
    s = min(units, max(1, (m - 1) * τ))
    if exact === nothing
        exact = widen(s) * m * m <= _CAMS_EXACT_WORK
    end
    P = zeros(Int, s, max(m, 1))
    R = zeros(Int, s + 1, m + 1)
    kind = zeros(Int8, s, max(m, 1))
    ks = zeros(Int32, s, max(m, 1))
    for j = 2:m
        R[1, j+1] = _CAMS_INF
    end
    @inbounds for i = 1:s
        hint0 = hint1 = 1
        for j = 2:m
            best = (j - 1) + P[i, j-1]
            bkind = _CAMS_NONE
            bk = 0
            if i >= 2
                v, k = _best_split(j - 1, exact, hint0) do k
                    k + P[i, k] + P[i-1, j-k]
                end
                hint0 = k
                if v < best
                    best, bkind, bk = v, _CAMS_STATE, k
                end
            end
            if i - 1 - τ >= 0
                v, k = _best_split(j - 1, exact, hint1) do k
                    (k - 1) + (k == 1 ? 0 : P[i, k-1]) + R[i-τ, j-k+1]
                end
                hint1 = k
                # On a tie the tape wins: it saves the copy of a state.
                if v <= best && v < _CAMS_INF
                    best, bkind, bk = v, _CAMS_TAPE, k
                end
            end
            P[i, j] = best
            kind[i, j] = bkind
            ks[i, j] = bk
        end
        for j = 2:m
            R[i+1, j+1] = min(P[i, j], i >= τ ? R[i-τ+1, j] : _CAMS_INF)
        end
    end
    return CAMSTable(m, s, τ, P, R, kind, ks, exact)
end

# Whether a segment of `j` steps with an unstored base and `i` free units
# starts by taping its first step rather than by storing the base.
@inline function _tape_first(t::CAMSTable, i::Int, j::Int)
    i >= t.stage_units || return false
    return i < 1 || t.R[i-t.stage_units+1, j] <= t.P[i, j]
end

"""
    cams_recomputations(steps, units; stage_units = 2)

The number of loop iterations a [`CAMS`](@ref) schedule runs forward in the
reverse pass, on top of the `steps` iterations of the loop itself. With
`units = 1` this is `steps * (steps - 1) / 2`; with enough units to tape every
step it is zero. [`Revolve`](@ref) with `c` checkpoints corresponds to
`units = c` and a `stage_units` too large to ever be used.
"""
function cams_recomputations(steps::Integer, units::Integer; stage_units::Integer = 2)
    t = CAMSTable(Int(steps), Int(units), Int(stage_units))
    return t.R[t.units+1, t.steps+1]
end

# --- the schedule as a list of actions --------------------------------------

@enum CAMSActionKind::Int8 begin
    cams_advance       # run steps a:b forward
    cams_store         # store the state in slot a
    cams_restore       # restore the state from slot a
    cams_tape          # run step a forward in split mode, keep its tape in slot b
    cams_adjoint       # adjoin step a from the state it starts from
    cams_adjoint_tape  # adjoin step a from the tape in slot b
end

struct CAMSAction
    kind::CAMSActionKind
    a::Int
    b::Int
end

# Adjoin steps base+1:base+j. The loop is at the state after `base` steps,
# which is in `baseslot`; `i` units are available including that one.
function _emit_stored!(acts, t::CAMSTable, base, j, i, baseslot, nslot, ntape)
    τ = t.stage_units
    while true
        if j == 1
            push!(acts, CAMSAction(cams_adjoint, base + 1, 0))
            return
        end
        kind = t.kind[i, j]
        k = Int(t.k[i, j])
        if kind == _CAMS_NONE
            push!(acts, CAMSAction(cams_advance, base + 1, base + j - 1))
            push!(acts, CAMSAction(cams_adjoint, base + j, 0))
            j -= 1
        elseif kind == _CAMS_STATE
            push!(acts, CAMSAction(cams_advance, base + 1, base + k))
            push!(acts, CAMSAction(cams_store, nslot, 0))
            _emit_stored!(acts, t, base + k, j - k, i - 1, nslot, nslot + 1, ntape)
            j = k
        else
            k > 1 && push!(acts, CAMSAction(cams_advance, base + 1, base + k - 1))
            push!(acts, CAMSAction(cams_tape, base + k, ntape))
            _emit_unstored!(acts, t, base + k, j - k, i - 1 - τ, nslot, ntape + 1)
            push!(acts, CAMSAction(cams_adjoint_tape, base + k, ntape))
            j = k - 1
            j == 0 && return
        end
        push!(acts, CAMSAction(cams_restore, baseslot, 0))
    end
end

# The same with the base state not stored and all `i` units free.
function _emit_unstored!(acts, t::CAMSTable, base, j, i, nslot, ntape)
    τ = t.stage_units
    first_tape = ntape
    first_step = base + 1
    while j >= 1
        if j == 1
            push!(acts, CAMSAction(cams_adjoint, base + 1, 0))
            break
        elseif _tape_first(t, i, j)
            push!(acts, CAMSAction(cams_tape, base + 1, ntape))
            ntape += 1
            base += 1
            j -= 1
            i -= τ
        else
            push!(acts, CAMSAction(cams_store, nslot, 0))
            _emit_stored!(acts, t, base, j, i, nslot, nslot + 1, ntape)
            break
        end
    end
    for n = (ntape-1):-1:first_tape
        push!(acts, CAMSAction(cams_adjoint_tape, first_step + n - first_tape, n))
    end
    return
end

function cams_actions(t::CAMSTable)
    acts = CAMSAction[]
    _emit_unstored!(acts, t, 0, t.steps, t.units, 1, 1)
    return acts
end

# --- the scheme -------------------------------------------------------------

mutable struct CAMS{FT} <: Scheme
    steps::Int
    units::Int
    stage_units::Int
    verbose::Int
    storage::AbstractStorage
    actions::Vector{CAMSAction}
end

"""
    CAMS(
        units::Integer;
        stage_units::Integer = 2,
        storage = ArrayStorage,
        verbose::Int = 0,
    )

Checkpointing for adjoint multistage schemes, for `for` loops.

[`Revolve`](@ref) stores states only, so each iteration runs forward one more
time when it is differentiated. `CAMS` can also store the *tape* of an
iteration, from which that iteration is differentiated without running it
again, and picks per checkpoint whichever of the two saves more. It never does
more recomputation than Revolve on the same memory, and with enough memory
it does none.

- `units`: the memory budget, in units of one state checkpoint.
- `stage_units`: the size of the tape of one iteration, in the same units.
- `storage`: the storage backend for the state checkpoints. Tapes always stay
  in memory.
- `verbose`: verbosity level.

This pays off when an iteration is expensive to run and cheap to differentiate
from its tape, as an implicit time step is whose nonlinear solve has a reverse
rule from the implicit function theorem: one iteration forward is a Newton
solve, its adjoint a single linear solve. For an explicit step differentiated
by Enzyme, running the step again is cheap next to its adjoint and Revolve is
the better choice.

The size of a tape is not visible to Julia, so `stage_units` has to be given;
it only steers the schedule and any value yields the correct gradient. Besides
the budget, the tape of the last iteration and one copy of the final state are
kept between the forward and the reverse pass. The schedule is tabulated ahead
of time, which takes `O(units * steps^2)` operations and `O(units * steps)`
memory; beyond a few thousand steps the search is narrowed and the schedule is
near-optimal rather than optimal. [`cams_recomputations`](@ref) predicts the
number of iterations recomputed.

# References

- Zhang, H. & Constantinescu, E. M. “Optimal Checkpointing for Adjoint
  Multistage Time-Stepping Schemes.” Journal of Computational Science 66 (2023).
"""
function CAMS(
    units::Integer;
    stage_units::Integer = 2,
    storage = ArrayStorage,
    verbose::Int = 0,
)
    units >= 1 || throw(ArgumentError("[Checkpointing.jl]: CAMS needs at least one unit."))
    stage_units >= 1 ||
        throw(ArgumentError("[Checkpointing.jl]: CAMS needs stage_units >= 1."))
    return CAMS{Nothing}(
        0,
        units,
        stage_units,
        verbose,
        _new_storage(storage, units),
        CAMSAction[],
    )
end

function instantiate(::Type{FT}, cams::CAMS{Nothing}, steps::Int) where {FT}
    table = CAMSTable(steps, cams.units, cams.stage_units)
    actions = cams_actions(table)
    if cams.verbose > 0
        extra = table.R[table.units+1, steps+1]
        @info "[Checkpointing] CAMS with $(cams.units) units, $(cams.stage_units) per tape"
        @info "[Checkpointing] Recomputed steps : $extra" *
              (table.exact ? "" : " (near-optimal schedule)")
        @info "[Checkpointing] Overhead factor  : $(steps == 0 ? 0.0 : extra / steps)"
    end
    return CAMS{FT}(
        steps,
        cams.units,
        cams.stage_units,
        cams.verbose,
        similar(cams.storage, FT),
        actions,
    )
end

function instantiate(::Type{FT}, ::CAMS{Nothing}) where {FT}
    throw(
        ArgumentError(
            "[Checkpointing.jl]: CAMS plans its schedule from the number of iterations " *
            "and supports `for` loops only. Use `Online_r2` for a `while` loop.",
        ),
    )
end
