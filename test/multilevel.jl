using Checkpointing
using Enzyme
using Test

mutable struct Chkp
    x::Vector{Float64}
    scheme::Scheme
end

function loops(chkp::Chkp, scheme1::Scheme, it1::Int, it2::Int)
    @checkpoint_struct scheme1 chkp for i in 1:it1
        @checkpoint_struct chkp.scheme chkp for j in 1:it2
            chkp.x .= 2.0 * sqrt.(chkp.x) .* sqrt.(chkp.x)
        end
    end
    return reduce(+, chkp.x)
end

it1 = 2
it2 = 5
periodic = Periodic{Chkp}(it1, 1)
revolve = Revolve{Chkp}(it2, 2)

x = Chkp([2.0, 3.0, 4.0], revolve)
dx = Chkp([0.0, 0.0, 0.0], revolve)

primal = loops(x, periodic, it1, it2)

peridoc = Periodic{Chkp}(it1, 1; verbose=0)
revolve = Revolve{Chkp}(it2, 2; verbose=0)

x = Chkp([2.0, 3.0, 4.0], revolve)
dx = Chkp([0.0, 0.0, 0.0], revolve)

g = autodiff(Enzyme.ReverseWithPrimal, loops, Active, Duplicated(x, dx), periodic, it1, it2)

# TODO: Primal is wrong only when multilevel checkpointing is used
@test_broken g[2] == primal
@test all(dx.x .== [1024.0, 1024.0, 1024.0])

