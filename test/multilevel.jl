using Checkpointing
using Enzyme
using Test

mutable struct Chkp
    x::Vector{Float64}
    scheme::Scheme
end

function loops(chkp::Chkp, scheme1::Scheme, it1::Int, it2::Int)
    @ad_checkpoint scheme1 for i = 1:it1
        @ad_checkpoint chkp.scheme for j = 1:it2
            chkp.x .= 2.0 * sqrt.(chkp.x) .* sqrt.(chkp.x)
        end
    end
    return reduce(+, chkp.x)
end

it1 = 2
it2 = 5
periodic = Periodic(1)
revolve = Revolve(2)

x = Chkp([2.0, 3.0, 4.0], revolve)
dx = Chkp([0.0, 0.0, 0.0], revolve)

primal = loops(x, periodic, it1, it2)

peridoc = Periodic(1; verbose = 0)
revolve = Revolve(2; verbose = 0)

x = Chkp([2.0, 3.0, 4.0], revolve)
dx = Chkp([0.0, 0.0, 0.0], revolve)

g = autodiff(
    Enzyme.ReverseWithPrimal,
    loops,
    Active,
    Duplicated(x, dx),
    Const(periodic),
    Const(it1),
    Const(it2),
)

# TODO: Primal is wrong only when multilevel checkpointing is used
@test g[2] == primal
@test all(dx.x .== [1024.0, 1024.0, 1024.0])
