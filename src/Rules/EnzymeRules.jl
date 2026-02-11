using Enzyme
import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules
import EnzymeCore

# Floating-point scalar types that can cause activity analysis issues when captured
# in closures alongside mutable structs. Integer types are always inactive in AD,
# so they don't cause the same problem.
const PROBLEMATIC_SCALAR_TYPES = Union{Float16,Float32,Float64}

"""
    check_closure_captures(body)

Check the closure for common issues that cause Enzyme activity analysis errors.
Provides helpful error messages pointing to the specific problematic variables.
"""
function check_closure_captures(body)
    closure_type = typeof(body.val)
    field_names = fieldnames(closure_type)
    field_types = fieldtypes(closure_type)

    # Find captured scalar variables and struct variables
    scalar_vars = String[]
    struct_vars = String[]

    for (name, ftype) in zip(field_names, field_types)
        field = getfield(body.val, name)
        if isa(field, Core.Box)
            error(
                "[Checkpointing.jl]: Variable `$name` is reassigned inside the loop. " *
                "Please make sure that `$name` is only modified in-place.",
            )
        elseif ftype <: PROBLEMATIC_SCALAR_TYPES
            push!(scalar_vars, string(name))
        elseif ismutabletype(ftype)
            push!(struct_vars, string(name))
        end
    end

    # If we have both floating-point scalars and mutable structs captured, warn the user.
    # This combination causes Enzyme activity analysis errors because Enzyme can't
    # determine whether the floating-point values should participate in AD.
    if !isempty(scalar_vars) && !isempty(struct_vars)
        scalar_list = join(["`$v`" for v in scalar_vars], ", ")
        struct_list = join(["`$v`" for v in struct_vars], ", ")
        error(
            "[Checkpointing.jl]: The loop body captures floating-point variable(s) $scalar_list " *
            "alongside mutable struct(s) $struct_list.\n" *
            "This causes Enzyme activity analysis errors.\n\n" *
            "Solution: Store these values as fields in your mutable struct instead of " *
            "capturing them as separate variables.\n\n" *
            "Example - instead of:\n" *
            "    h = 0.1\n" *
            "    @ad_checkpoint scheme for i in 1:n\n" *
            "        model.t += h  # ERROR: h captured from outer scope\n" *
            "    end\n\n" *
            "Use:\n" *
            "    model.h = 0.1  # store in struct\n" *
            "    @ad_checkpoint scheme for i in 1:n\n" *
            "        model.t += model.h  # OK: access through struct\n" *
            "    end"
        )
    end
end

function augmented_primal(
    config,
    func::Const{typeof(Checkpointing.checkpoint_for)},
    ret,
    body,
    alg,
    range,
)
    check_closure_captures(body)
    tape_body = deepcopy(body.val)
    # make_zero!(body.dval)
    func.val(body.val, alg.val, range.val)
    if needs_primal(config)
        return AugmentedReturn(nothing, nothing, (tape_body,))
    else
        return AugmentedReturn(nothing, nothing, (tape_body,))
    end
end

function reverse(
    config,
    ::Const{typeof(Checkpointing.checkpoint_for)},
    dret::Type{<:Const},
    tape,
    body::Union{Const,Duplicated,MixedDuplicated},
    alg,
    range,
)
    (body_input,) = tape
    scheme = instantiate(typeof(body_input), alg.val, length(range.val))
    dbody = if isa(body, Duplicated)
        body.dval
    elseif isa(body, MixedDuplicated)
        body.dval[]
    elseif isa(body, Const)
        # This happens when the closure has mixed activity (e.g., captures both
        # a mutable struct and scalar variables). Provide a helpful error.
        check_closure_captures(body)
        # If check_closure_captures didn't error, give a generic message
        error(
            "[Checkpointing.jl]: The loop body was marked as Const by Enzyme, " *
            "but checkpointing requires an active (Duplicated) closure. " *
            "Make sure your loop body captures a mutable struct that is being differentiated."
        )
    else
        error("Checkpointing.jl: Unknown annotation type for body: $(typeof(body))")
    end

    Checkpointing.rev_checkpoint_for(config, body_input, dbody, scheme, range.val)
    return (nothing, nothing, nothing)
end

function augmented_primal(
    config,
    func::Const{typeof(Checkpointing.checkpoint_while)},
    ret,
    body,
    alg,
)
    check_closure_captures(body)
    tape_body = deepcopy(body.val)
    # make_zero!(body.dval)
    func.val(body.val, alg.val)
    if needs_primal(config)
        return AugmentedReturn(nothing, nothing, (tape_body,))
    else
        return AugmentedReturn(nothing, nothing, (tape_body,))
    end
end

function reverse(
    config,
    ::Const{typeof(Checkpointing.checkpoint_while)},
    dret::Type{<:Const},
    tape,
    body::Union{Const,Duplicated,MixedDuplicated},
    alg,
)
    (body_input,) = tape
    scheme = instantiate(typeof(body_input), alg.val)
    dbody = if isa(body, Duplicated)
        body.dval
    elseif isa(body, MixedDuplicated)
        body.dval[]
    elseif isa(body, Const)
        # This happens when the closure has mixed activity (e.g., captures both
        # a mutable struct and scalar variables). Provide a helpful error.
        check_closure_captures(body)
        # If check_closure_captures didn't error, give a generic message
        error(
            "[Checkpointing.jl]: The loop body was marked as Const by Enzyme, " *
            "but checkpointing requires an active (Duplicated) closure. " *
            "Make sure your loop body captures a mutable struct that is being differentiated."
        )
    else
        error("Checkpointing.jl: Unknown annotation type for body: $(typeof(body))")
    end

    Checkpointing.rev_checkpoint_while(config, body_input, dbody, scheme)
    return (nothing, nothing)
end

export augmented_primal, reverse
