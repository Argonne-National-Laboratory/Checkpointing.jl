using Checkpointing

function main(steps, checkpoints; verbose=0)
    store = function f() end
    revolve = Revolve(steps, checkpoints, store, store; verbose=verbose)
    guessed_checkpoints = guess(revolve)
    println("Revolve suggests : $guessed_checkpoints checkpoints for a factor of $(factor(steps,guessed_checkpoints))")
    println("actions: ")
    while true
        next_action = next_action!(revolve)
        if next_action.actionflag == Checkpointing.done
            break
        end
    end
    return revolve
end