module Checkpointing

using LinearAlgebra
using Parameters

export mynorm

abstract type Scheme end

include("Revolve/Revolve.jl")
#include("Revolve/Periodic.jl")

export Revolve, guess, factor, next_action!, ActionFlag

end
