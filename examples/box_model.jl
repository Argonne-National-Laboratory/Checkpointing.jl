using Adapt

const blength = [5000.0e5; 1000.0e5; 5000.0e5]   ## north-south size of boxes, centimeters

const bdepth = [1.0e5; 5.0e5; 4.0e5]   ## depth of boxes, centimeters

const delta = bdepth[1] / (bdepth[1] + bdepth[3])  ## constant ratio of two depths

const bwidth = 4000.0 * 1e5  ## box width, centimeters

# box areas
const barea = [
    blength[1] * bwidth
    blength[2] * bwidth
    blength[3] * bwidth
]

# box volumes
const bvol = [
    barea[1] * bdepth[1]
    barea[2] * bdepth[2]
    barea[3] * bdepth[3]
]

# parameters that are used to ensure units are in CGS (cent-gram-sec)

const hundred = 100.0
const thousand = 1000.0
const day = 3600.0 * 24.0
const year = day * 365.0
const Sv = 1e12     ## one Sverdrup (a unit of ocean transport), 1e6 meters^3/second

# parameters that appear in box model equations
const u0 = 16.0 * Sv / 0.0004
const alpha = 1668e-7
const beta = 0.7811e-3

const gamma = 1 / (300 * day)

# robert filter coefficient for the smoother part of the timestep
const robert_filter_coeff = 0.25

# freshwater forcing
const FW = [(hundred / year) * 35.0 * barea[1]; -(hundred / year) * 35.0 * barea[1]]

# restoring atmospheric temperatures
const Tstar = [22.0; 0.0]
const Sstar = [36.0; 34.0];

# function to compute transport
#       Input: rho - the density vector
#       Output: U - transport value

function U_func(dens)

    U = u0 * (dens[2] - (delta * dens[1] + (1 - delta) * dens[3]))
    return U

end

# function to compute density
#       Input: state = [T1; T2; T3; S1; S2; S3]
#       Output: rho

function rho_func(state)
    T = @view state[1:3]
    S = @view state[4:6]
    return .-alpha .* T .+ beta .* S
end

# lastly our timestep function
#       Input: fld_now = [T1(t), T2(t), ..., S3(t)]
#           fld_old = [T1(t-dt), ..., S3(t-dt)]
#           u = transport(t)
#           dt = time step
#       Output: fld_new = [T1(t+dt), ..., S3(t+dt)]

function timestep_func(fld_now, fld_old, u, dt)

    temp = zero(fld_now)

    # first computing the time derivatives of the various temperatures and salinities
    if u > 0

        temp[1] = u * (fld_now[3] - fld_now[1]) / bvol[1] + gamma * (Tstar[1] - fld_now[1])
        temp[2] = u * (fld_now[1] - fld_now[2]) / bvol[2] + gamma * (Tstar[2] - fld_now[2])
        temp[3] = u * (fld_now[2] - fld_now[3]) / bvol[3]

        temp[4] = u * (fld_now[6] - fld_now[4]) / bvol[1] + FW[1] / bvol[1]
        temp[5] = u * (fld_now[4] - fld_now[5]) / bvol[2] + FW[2] / bvol[2]
        temp[6] = u * (fld_now[5] - fld_now[6]) / bvol[3]

    elseif u <= 0

        temp[1] = u * (fld_now[2] - fld_now[1]) / bvol[1] + gamma * (Tstar[1] - fld_now[1])
        temp[2] = u * (fld_now[3] - fld_now[2]) / bvol[2] + gamma * (Tstar[2] - fld_now[2])
        temp[3] = u * (fld_now[1] - fld_now[3]) / bvol[3]

        temp[4] = u * (fld_now[5] - fld_now[4]) / bvol[1] + FW[1] / bvol[1]
        temp[5] = u * (fld_now[6] - fld_now[5]) / bvol[2] + FW[2] / bvol[2]
        temp[6] = u * (fld_now[4] - fld_now[6]) / bvol[3]

    end

    # update fldnew using a version of Euler's method
    fld_new = fld_old .+ 2.0 .* dt .* temp

    return fld_new
end

mutable struct Box{T}
    in_now::T
    in_old::T
    out_now::T
    out_old::T
    i::Int
end

function Adapt.adapt_structure(to, box::Box)
    Box(adapt(to, box.in_now), adapt(to, box.in_old),
        adapt(to, box.out_now), adapt(to, box.out_old), box.i)
end

function forward_func_4_AD(in_now, in_old, out_old, out_now)
    rho_now = rho_func(in_now)                             ## compute density
    u_now = U_func(rho_now)                                ## compute transport
    in_new = timestep_func(in_now, in_old, u_now, 10 * day)  ## compute new state values
    in_now .= in_now .+ robert_filter_coeff .* (in_new .- 2.0 .* in_now .+ in_old)
    out_old .= in_now
    out_now .= in_new
    return nothing
end


function advance(box::Box)
    forward_func_4_AD(box.in_now, box.in_old, box.out_now, box.out_old)
end

function timestepper(box::Box, scheme::Union{Revolve,Periodic}, tsteps::Int)
    @ad_checkpoint scheme for i = 1:tsteps
        advance(box)
        box.in_now .= box.out_old
        box.in_old .= box.out_now
    end
    return box.out_now[1]
end

function timestepper(box::Box, scheme::Union{Online_r2}, tsteps::Int)
    box.i = 1
    @ad_checkpoint scheme while box.i <= tsteps
        advance(box)
        box.in_now .= box.out_old
        box.in_old .= box.out_now
        box.i += one(box.i)
    end
    return box.out_now[1]
end


function box(scheme::Scheme, tsteps::Int)
    Tbar = [20.0; 1.0; 1.0]
    Sbar = [35.5; 34.5; 34.5]

    # Create object from struct. tsteps is not needed for a for-loop
    box = Box(copy([Tbar; Sbar]), copy([Tbar; Sbar]), zeros(6), zeros(6), 0)
    dbox = Box(zeros(6), zeros(6), zeros(6), zeros(6), 0)

    # Compute gradient
    autodiff(
        Enzyme.ReverseWithPrimal,
        Const(timestepper),
        Duplicated(box, dbox),
        Const(scheme),
        Const(tsteps),
    )
    return box.out_now[1], dbox.in_old
end
