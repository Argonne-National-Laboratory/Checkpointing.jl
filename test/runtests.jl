using Test
using Checkpointing
using LinearAlgebra

@testset "Testing Revolve..." begin
	global steps = 50
	global checkpoints = 7
	global verbose = 0
	include("../examples/printaction.jl")
	revolve = main(steps, checkpoints)

	@test revolve.numfwd == 105
	@test revolve.numstore == 28
	@test revolve.numinv == 177
end
