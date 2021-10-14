using Checkpointing
using LinearAlgebra


function mynorm(x)
	return sqrt(dot(x,x))
end

mynorm(x)