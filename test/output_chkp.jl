using Checkpointing
using Enzyme
using Serialization
using Test

mutable struct ChkpOut
    x::Vector{Float64}
end

function loops(chkp::ChkpOut, scheme::Scheme, iters::Int)
    @checkpoint_struct scheme chkp for i in 1:iters
        chkp.x .= 2.0 * sqrt.(chkp.x) .* sqrt.(chkp.x)
    end
    return reduce(+, chkp.x)
end
iters = 10
# Checkpoint every 2nd timestep
revolve = Revolve{ChkpOut}(
    iters,
    3;
    verbose=0,
    write_checkpoints=true,
    write_checkpoints_filename="chkp",
    write_checkpoints_period=2
)

x = ChkpOut([2.0, 3.0, 4.0])
dx = ChkpOut([0.0, 0.0, 0.0])

g = autodiff(Enzyme.Reverse, loops, Active, Duplicated(x, dx), revolve, iters)

fid = Checkpointing.HDF5.h5open("adjoint_chkp.h5", "r")
# List all checkpoints
saved_chkp = sort(parse.(Int, (keys(fid))))
println("Checkpoints saved: $saved_chkp")
chkp = Checkpointing.deserialize(read(fid["3"]))
@test isa(chkp, ChkpOut)
close(fid)
