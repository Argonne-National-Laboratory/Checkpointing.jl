using Checkpointing

function main(steps, checkpoints; verbose = 0)
    revolve = Revolve{Nothing}(steps, checkpoints; verbose = verbose)
    while true
        next_action = next_action!(revolve)
        if next_action.actionflag == Checkpointing.done
            break
        end
    end
    return revolve
end
