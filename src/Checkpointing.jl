module Checkpointing

using LinearAlgebra

export mynorm

abstract type Scheme end

include("Revolve/Revolve.jl")

export Revolve
export init

function mynorm(x)
	return sqrt(dot(x,x))
end

greet() = print("Hello World!")

end # module
