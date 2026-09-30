# Checkpointing.jl's schemes scheduling a loop that Enzyme's LLVM core
# differentiates (through enzyme/checkpoint.h). Needs a clang and the matching
# ClangEnzyme plugin: set ENZYME_CLANG and ENZYME_CLANG_PLUGIN.
module EnzymeABITest

using Checkpointing
using HDF5
using Test

const CLANG = get(ENV, "ENZYME_CLANG", "")
const PLUGIN = get(ENV, "ENZYME_CLANG_PLUGIN", "")

if isempty(CLANG) || isempty(PLUGIN)
    @info "Skipping the Enzyme C interface tests: set ENZYME_CLANG and ENZYME_CLANG_PLUGIN"
else
    lib = joinpath(mktempdir(), "libloop.so")
    run(`$CLANG -O2 -shared -fPIC -fpass-plugin=$PLUGIN -Xclang -load -Xclang $PLUGIN
         $(joinpath(@__DIR__, "enzyme_abi", "loop.c")) -o $lib -lm`)

    n_state = ccall((:state_size, lib), Int64, ())
    u0 = [1.0 + 0.5 * sin(k) for k = 0:(n_state-1)]

    function reference(n)
        u = copy(u0)
        du = zeros(n_state)
        ccall((:gradient_plain, lib), Cvoid, (Ptr{Float64}, Ptr{Float64}, Int64), u, du, n)
        return du
    end

    function checkpointed(alg, n)
        u = copy(u0)
        du = zeros(n_state)
        scheme, data = enzyme_scheme(alg)
        GC.@preserve alg ccall(
            (:gradient_checkpointed, lib),
            Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Int64, Ptr{Cvoid}, Ptr{Cvoid}),
            u,
            du,
            n,
            scheme,
            data,
        )
        return du
    end

    @testset "Enzyme C interface: $name" for (name, alg) in [
        ("Revolve(1)", () -> Revolve(1)),
        ("Revolve(3)", () -> Revolve(3)),
        ("Periodic(2)", () -> Periodic(2)),
        ("Periodic(5)", () -> Periodic(5)),
        ("Revolve(3) in HDF5", () -> Revolve(3; storage = HDF5Storage)),
        ("Periodic(3) in HDF5", () -> Periodic(3; storage = HDF5Storage)),
    ]
        for n in (1, 2, 7, 30)
            @test checkpointed(alg(), n) ≈ reference(n) rtol = 1e-12
        end
    end
    @test isempty(Checkpointing.LIVE_SCHEDULES)
end

end # module EnzymeABITest
