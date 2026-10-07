using Test
using InteractiveUtils: code_llvm
using CheckpointingCore
using CheckpointingCore: checkpoint_schedule, loop_checkpoint

function marked(x, n)
    y = x
    @ad_checkpoint Revolve(3) for i in 1:n
        y = sin(y) + 0.1 * i
    end
    return y
end

function plain(x, n)
    y = x
    for i in 1:n
        y = sin(y) + 0.1 * i
    end
    return y
end

@testset "CheckpointingCore" begin
    @test checkpoint_schedule(:(Revolve(3))) == (:revolve, 3)
    @test checkpoint_schedule(:(Checkpointing.Periodic(10))) == (:periodic, 10)
    @test checkpoint_schedule(:(Binomial(2))) == (:binomial, 2)
    @test checkpoint_schedule(:(StoreAll())) == (:store_all, 0)
    @test checkpoint_schedule(:(Revolve())) == (:revolve, 0)
    @test checkpoint_schedule(:(Revolve(k))) === nothing
    @test checkpoint_schedule(:(Revolve(3; verbose = 1))) === nothing
    @test checkpoint_schedule(:scheme) === nothing

    expand(s) = macroexpand(@__MODULE__, Meta.parse(s))
    ex = expand("@ad_checkpoint Periodic(4) for t in 1:n; step!(m); end")
    @test loop_checkpoint(ex) == (:periodic, 4)
    ex = expand("@ad_checkpoint Binomial(5) while t < n; t += 1; end")
    @test ex.head === :while
    @test loop_checkpoint(ex) == (:binomial, 5)
    @test_throws Exception expand("@ad_checkpoint scheme for i in 1:n; end")

    # The annotation changes nothing about running the loop.
    @test marked(0.3, 10) == plain(0.3, 10)
    ir = sprint(io -> code_llvm(io, marked, (Float64, Int); raw = true, dump_module = true))
    @test occursin("!{!\"enzyme.checkpoint\", !\"revolve\", i64 3}", ir)
end
