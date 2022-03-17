using Checkpointing
using LinearAlgebra
using Enzyme
using Test

include("examples/mutable/optcontrol.jl")

function chkmutable()
    global steps = 100
    global snaps = 3
    global info = 0

    revolve = Revolve(steps, snaps; verbose=info)

    model, shadowmodel = muoptcontrol(revolve, steps)
    return model.F, shadowmodel.F
end

mF, mL = chkmutable()
