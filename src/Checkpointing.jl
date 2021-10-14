module Checkpointing

using LinearAlgebra

export mynorm

function mynorm(x)
	return sqrt(dot(x,x))
end

greet() = print("Hello World!")

end # module
