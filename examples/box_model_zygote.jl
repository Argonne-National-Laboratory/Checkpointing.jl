const blength = [5000.0e5; 1000.0e5; 5000.0e5]   ## north-south size of boxes, centimeters

const bdepth = [1.0e5; 5.0e5; 4.0e5]   ## depth of boxes, centimeters

const delta = bdepth[1]/(bdepth[1] + bdepth[3])  ## constant ratio of two depths

const bwidth = 4000.0*1e5  ## box width, centimeters

# box areas
const barea = [blength[1]*bwidth;
         blength[2]*bwidth;
         blength[3]*bwidth]

# box volumes
const bvol = [barea[1]*bdepth[1];
        barea[2]*bdepth[2];
        barea[3]*bdepth[3]]

# parameters that are used to ensure units are in CGS (cent-gram-sec)

const hundred = 100.0
const thousand = 1000.0
const day = 3600.0*24.0
const year = day*365.0
const Sv = 1e12     ## one Sverdrup (a unit of ocean transport), 1e6 meters^3/second

# parameters that appear in box model equations
const u0 = 16.0*Sv/0.0004
const alpha = 1668e-7
const beta = 0.7811e-3

const gamma = 1/(300*day)

# robert filter coefficient for the smoother part of the timestep
const robert_filter_coeff = 0.25

# freshwater forcing
const FW = [(hundred/year) * 35.0 * barea[1]; -(hundred/year) * 35.0 * barea[1]]

# restoring atmospheric temperatures
const Tstar = [22.0; 0.0]
const Sstar = [36.0; 34.0];

# function to compute transport
#       Input: rho - the density vector
#       Output: U - transport value

function U_func(dens)

    U = u0*(dens[2] - (delta * dens[1] + (1 - delta)*dens[3]))
    return U

end

# function to compute density
#       Input: state = [T1; T2; T3; S1; S2; S3]
#       Output: rho

function rho_func(state)

    rho = zeros(3)

    rho[1] = -alpha * state[1] + beta * state[4]
    rho[2] = -alpha * state[2] + beta * state[5]
    rho[3] = -alpha * state[3] + beta * state[6]

    return rho

end

# lastly our timestep function
#       Input: fld_now = [T1(t), T2(t), ..., S3(t)]
#           fld_old = [T1(t-dt), ..., S3(t-dt)]
#           u = transport(t)
#           dt = time step
#       Output: fld_new = [T1(t+dt), ..., S3(t+dt)]

function timestep_func(fld_now, fld_old, u, dt)

    temp = zeros(6)
    fld_new = zeros(6)

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

    for j = 1:6
        fld_new[j] = fld_old[j] + 2.0 * dt * temp[j]
    end

    return fld_new
end

mutable struct Box
    in_now::Vector{Float64}
    in_old::Vector{Float64}
    out_now::Vector{Float64}
    out_old::Vector{Float64}
    i::Int
end

function forward_func_4_AD(in_now, in_old, out_old, out_now)
    rho_now = rho_func(in_now)                             ## compute density
    u_now = U_func(rho_now)                                ## compute transport
    in_new = timestep_func(in_now, in_old, u_now, 10*day)  ## compute new state values
    for j = 1:6
        in_now[j] = in_now[j] + robert_filter_coeff * (in_new[j] - 2.0 * in_now[j] + in_old[j])
    end
    out_old[:] = in_now
    out_now[:] = in_new
    return nothing
end


function advance(box::Box)
    forward_func_4_AD(box.in_now, box.in_old, box.out_now, box.out_old)
end

function timestepper_for(box::Box, scheme::Scheme, tsteps::Int)
    @checkpoint_struct scheme box for i in 1:tsteps
        advance(box)
        box.in_now[:] = box.out_old
        box.in_old[:] = box.out_now
        nothing
    end
    return box.out_now[1]
end


function box_for(scheme::Scheme, tsteps::Int)
    Tbar = [20.0; 1.0; 1.0]
    Sbar = [35.5; 34.5; 34.5]

    # Create object from struct. tsteps is not needed for a for-loop
    box = Box(copy([Tbar; Sbar]), copy([Tbar; Sbar]), zeros(6), zeros(6), 0)

    # Compute gradient
    g = Zygote.gradient(timestepper_for, box, scheme, tsteps)
    return box.out_now[1], g[1][2]
end

function timestepper_while(box::Box, scheme::Scheme, tsteps::Int)
    box.i=1
    @checkpoint_struct scheme box while box.i <= tsteps
        advance(box)
        box.in_now[:] = box.out_old
        box.in_old[:] = box.out_now
        box.i = box.i+1
        nothing
    end
    return box.out_now[1]
end


function box_while(scheme::Scheme, tsteps::Int)
    Tbar = [20.0; 1.0; 1.0]
    Sbar = [35.5; 34.5; 34.5]

    # Create object from struct. tsteps is not needed for a for-loop
    box = Box(copy([Tbar; Sbar]), copy([Tbar; Sbar]), zeros(6), zeros(6), 0)

    # Compute gradient
    g = Zygote.gradient(timestepper_while, box, scheme, tsteps)
    return box.out_now[1], g[1][2]
end
