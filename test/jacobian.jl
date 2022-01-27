using Checkpointing
using ReverseDiff
using Zygote
using ForwardDiff
using Enzyme
using Test

function f(x)
    return [(x[1]*(x[2]-2.0))^2, (x[1]-1.0)^2*x[2]^2]
end
x = [2.0,6.0]

J_Re = Checkpointing.jacobian(f, x, ReverseDiffADTool())
J_Zy = Checkpointing.jacobian(f, x, ZygoteADTool())
J_Fo = Checkpointing.jacobian(f, x, ForwardDiffADTool())
J_En = Checkpointing.jacobian(f, x, EnzymeADTool())
# J_Di = Checkpointing.jacobian(f, x, DiffractorADTool())

@test J_Re ≈ J_Zy ≈ J_Fo ≈ J_En
