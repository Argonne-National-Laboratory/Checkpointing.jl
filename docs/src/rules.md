# Rules

Currently, EnzymeRules and ChainRulesCore rules are provided for checkpointing. Adding a new rule system support for checkpointing requires calling the augmented forward run for the while and for loops
```julia
function fwd_checkpoint_struct_for(body::Function, scheme::Scheme, model, range::UnitRange{Int64}) end
function fwd_checkpoint_struct_while(body::Function, scheme::Scheme, model, condition::Function) end
```
and the reverse call, respectively.
```julia
function rev_checkpoint_struct_for(
    body::Function,
    alg::Scheme,
    model_input::MT,
    shadowmodel::MT,
    range::UnitRange
) where {MT} end
function rev_checkpoint_struct_while(
    body::Function,
    alg::Online_r2,
    model_input::MT,
    shadowmodel::MT,
    condition::Function
) where {MT} end
```
