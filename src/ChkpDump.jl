struct ChkpDump
    steps::Int64
    period::Int64
    filename::String
end

ChkpDump(steps, ::Val{false}, period = 1, filename = "chkp") = nothing

function ChkpDump(steps, ::Val{true}, period = 1, filename = "chkp")
    return ChkpDump(steps, period, filename)
end

dump_prim(::Nothing, _, _) = nothing

function dump_prim(chkp::ChkpDump, step, primal)
    if (step - 1) % chkp.period == 0
        blob = serialize(primal)
        open("prim_$(chkp.filename)_$step.chkp", "w") do file
            write(file, blob)
        end
    end
end

dump_adj(::Nothing, _, _) = nothing

function dump_adj(chkp::ChkpDump, step, adjoint)
    @show step
    @show chkp.period
    @show step % chkp.period
    if (step - 1) % chkp.period == 0
        blob = serialize(adjoint)
        open("adj_$(chkp.filename)_$step.chkp", "w") do file
            write(file, blob)
        end
    end
end

read_chkp_file(filename) = deserialize(read(filename))