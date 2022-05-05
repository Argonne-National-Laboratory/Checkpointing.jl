

# Explicit 1D heat equation

```@example heat
using Checkpointing
using Plots
using Zygote

mutable struct Heat
    Tnext::Vector{Float64}
    Tlast::Vector{Float64}
    n::Int
    λ::Float64
    tsteps::Int
end

function advance(heat)
    next = heat.Tnext
    last = heat.Tlast
    λ = heat.λ
    n = heat.n
    for i in 2:(n-1)
        next[i] = last[i] + λ*(last[i-1]-2*last[i]+last[i+1])
    end
    return nothing
end


function sumheat(heat::Heat)
    # AD: Create shadow copy for derivatives 
    shadowheat = Heat(zeros(n), zeros(n), 0, 0.0, 0)
    @checkpoint_struct revolve heat shadowheat for i in 1:tsteps
        heat.Tlast .= heat.Tnext
        advance(heat)
    end
    return reduce(+, heat.Tnext)
end

n = 100
Δx=0.1
Δt=0.001
# Select μ such that λ ≤ 0.5 for stability with μ = (λ*Δt)/Δx^2
λ = 0.5
# time steps
tsteps = 500

# Create object from struct
heat = Heat(zeros(n), zeros(n), n, λ, tsteps)

# Boundary conditions
heat.Tnext[1]   = 20.0
heat.Tnext[end] = 0

# Set up AD
# Number of available snapshots
snaps = 4
verbose = 0
revolve = Revolve(tsteps, snaps; verbose=verbose)

# Compute gradient
g = Zygote.gradient(sumheat,heat)
```

Plot function values:
```@example heat
plot(heat.Tnext)
```
Plot gradient with respect to sum(T):
```@example heat
plot(g[1].Tnext[2:end-1])
```