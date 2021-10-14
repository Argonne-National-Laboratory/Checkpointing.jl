using Test
using Checkpointing
using LinearAlgebra

x = zeros(10)
x .= 2.0

@test norm(x) ≈ mynorm(x)


