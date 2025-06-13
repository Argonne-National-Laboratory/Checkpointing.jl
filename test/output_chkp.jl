using Checkpointing
using Enzyme
using Serialization
using Test
import Base.length
import Base.iterate
mutable struct ChkpOut
    x::Vector{Float64}
end

function loops(chkp::ChkpOut, scheme::Scheme, iters::Int)
    @ad_checkpoint scheme for i = 1:iters
        chkp.x .= 2.0 * sqrt.(chkp.x) .* sqrt.(chkp.x)
    end
    return reduce(+, chkp.x)
end

function output_chkp(scheme)
    iters = 10
    # Checkpoint every 2nd timestep
    _scheme = eval(scheme)(
        3;
        verbose = 0,
        write_checkpoints = true,
        write_checkpoints_filename = "chkp",
        write_checkpoints_period = 2,
    )

    x = ChkpOut([2.0, 3.0, 4.0])
    dx = ChkpOut([0.0, 0.0, 0.0])

    g = autodiff(
        Enzyme.Reverse,
        loops,
        Active,
        Duplicated(x, dx),
        Const(_scheme),
        Const(iters),
    )

    blob = Checkpointing.deserialize(read("adj_chkp_1.chkp"))
    # List all checkpoints
    @test all(dx.x .== blob.chkp.x)
end
