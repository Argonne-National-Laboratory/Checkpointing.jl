using Checkpointing

function main(steps, checkpoints; verbose = 0)
    revolve = Revolve(checkpoints; verbose = verbose)
    revolve = instantiate(Nothing, revolve, steps)
    while true
        next_action = Checkpointing.next_action!(revolve)
        if next_action.actionflag == Checkpointing.done
            break
        end
    end
    return revolve
end
