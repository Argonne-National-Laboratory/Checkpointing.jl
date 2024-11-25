using Checkpointing
using Enzyme
using Serialization
using Test
import Base.length
import Base.iterate
mutable struct ChkpOut
    x::Vector{Float64}
end

Base.length(chkp::ChkpOut) = length(chkp.x)
Base.iterate(chkp::ChkpOut) = iterate(chkp.x)
Base.iterate(chkp::ChkpOut, i) = iterate(chkp.x, i)

function loops(chkp::ChkpOut, scheme::Scheme, iters::Int)
    @checkpoint_struct scheme chkp for i = 1:iters
        chkp.x .= 2.0 * sqrt.(chkp.x) .* sqrt.(chkp.x)
    end
    return reduce(+, chkp.x)
end
iters = 10
# Checkpoint every 2nd timestep
revolve = Revolve{ChkpOut}(
    iters,
    3;
    verbose = 0,
    write_checkpoints = true,
    write_checkpoints_filename = "chkp",
    write_checkpoints_period = 2,
)

x = ChkpOut([2.0, 3.0, 4.0])
dx = ChkpOut([0.0, 0.0, 0.0])

g = autodiff(Enzyme.Reverse, loops, Active, Duplicated(x, dx), Const(revolve), Const(iters))

chkp = Checkpointing.deserialize(read("adj_chkp_1.chkp"))
# List all checkpoints
@test isa(chkp, ChkpOut)
@test all(dx .== chkp.x)
