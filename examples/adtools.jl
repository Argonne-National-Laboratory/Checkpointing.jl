using Checkpointing

struct ReverseDiffADTool <: AbstractADTool end
struct ZygoteADTool <: AbstractADTool end
struct DiffractorADTool <: AbstractADTool end
struct EnzymeADTool <: AbstractADTool end
struct ForwardDiffADTool <: AbstractADTool end

function Checkpointing.jacobian(tobedifferentiated, F_H, ::ReverseDiffADTool)
    return ReverseDiff.jacobian(tobedifferentiated, F_H)
end

function Checkpointing.jacobian(tobedifferentiated, F_H, ::ZygoteADTool)
    return Zygote.jacobian(tobedifferentiated, F_H)[1]
end

function Checkpointing.jacobian(tobedifferentiated, F_H, ::ForwardDiffADTool)
    return ForwardDiff.jacobian(tobedifferentiated, F_H)
end

function Checkpointing.jacobian(tobedifferentiated, F_H, ::EnzymeADTool)
    function f(x,res)
        y = tobedifferentiated(x)
        copyto!(res,y)
        return nothing
    end
    J = zeros(eltype(F_H), length(F_H), length(F_H))
    x = zeros(eltype(F_H), length(F_H))
    dx = zeros(eltype(F_H), length(F_H))
    y = zeros(eltype(F_H), length(F_H))
    dy = zeros(eltype(F_H), length(F_H))
    for i in 1:length(F_H)
        copyto!(x, F_H)
        fill!(dx, 0)
        fill!(y, 0)
        dy[i] = 1.0
        autodiff(f, Duplicated(x,dx), Duplicated(y, dy))
        J[i,:] = dx[:]
    end
    return J
end

function Checkpointing.jacobian(tobedifferentiated, F_H, ::DiffractorADTool)
    J = zeros(eltype(F_H), length(F_H), length(F_H))
    for i in 1:length(F_H)
        grad = Diffractor.gradient(x -> tobedifferentiated(x)[i], F_H)
        J[i,:] = grad[:][1]
    end
    return J
end