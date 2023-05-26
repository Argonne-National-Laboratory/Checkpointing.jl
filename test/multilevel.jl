using Checkpointing
using Enzyme
using Test

mutable struct Chkp
    x::Vector{Float64}
end

function loops(chkp::Chkp, scheme1::Scheme, scheme2::Scheme, it1::Int64, it2::Int64)
    @checkpoint_struct scheme1 chkp for i in 1:it1
        @checkpoint_struct scheme2 chkp for j in 1:it2
            chkp.x .= 2.0 * sqrt.(chkp.x) .* sqrt.(chkp.x)
        end
    end
    return reduce(+, chkp.x)
end

x = Chkp([2.0, 3.0, 4.0])
dx = Chkp([0.0, 0.0, 0.0])

revolve = Revolve{Chkp}(5, 2)
periodic = Periodic{Chkp}(2, 2)

primal = loops(x, revolve, periodic, 10,  1)

x = Chkp([2.0, 3.0, 4.0])
dx = Chkp([0.0, 0.0, 0.0])
revolve = Revolve{Chkp}(5, 2)
periodic = Periodic{Chkp}(2, 2; storage=HDF5Storage{Chkp}(2))
g = autodiff(Enzyme.ReverseWithPrimal, loops, Duplicated(x, dx), periodic, revolve, 2, 5)

@test g[2] == primal
@test all(dx.x .== [1024.0, 1024.0, 1024.0])

