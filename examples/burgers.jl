# Viscous Burgers equation u_t + (u²/2)_x = ν u_xx on the periodic unit interval.
# The crest of a Gaussian bump travels fastest, so the wave steepens into a shock.
using Checkpointing
using Enzyme

mutable struct Burgers
    u::Vector{Float64}      # state at the current time step
    ulast::Vector{Float64}  # state at the previous time step
    ν::Float64              # viscosity
    Δx::Float64
    Δt::Float64
end

Burgers(u0; ν = 0.005, Δt = 1e-3) = Burgers(copy(u0), similar(u0), ν, 1 / length(u0), Δt)

# One explicit Euler step, central differences, periodic boundaries.
function step!(b::Burgers)
    b.ulast .= b.u
    u, n = b.ulast, length(b.u)
    for i = 1:n
        l = i == 1 ? n : i - 1
        r = i == n ? 1 : i + 1
        flux = (u[r]^2 - u[l]^2) / (4 * b.Δx)
        diffusion = b.ν * (u[r] - 2 * u[i] + u[l]) / b.Δx^2
        b.u[i] = u[i] + b.Δt * (diffusion - flux)
    end
    return nothing
end

# u(x_k, T) after `steps` time steps.
function probe(b::Burgers, scheme::Scheme, steps::Int, k::Int)
    @ad_checkpoint scheme for i = 1:steps
        step!(b)
    end
    return b.u[k]
end

# The gradient of u(x_k, T) with respect to the initial condition u(x, 0).
function sensitivity(u0::Vector{Float64}, scheme::Scheme, steps::Int, k::Int)
    b = Burgers(u0)
    db = Enzyme.make_zero(b)
    autodiff(Reverse, probe, Duplicated(b, db), Const(scheme), Const(steps), Const(k))
    return db.u
end
