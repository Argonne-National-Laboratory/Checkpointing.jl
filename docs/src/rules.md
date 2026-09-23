# Rules

Currently, EnzymeRules rules are provided for checkpointing. The `augmented_primal` rules for `checkpoint_for` and `checkpoint_while` instantiate the scheme and run its forward half, `fwd_checkpoint_for` or `fwd_checkpoint_while`, which computes the primal and stores the checkpoints. The `reverse` rules pass the resulting tape to the scheme's reverse half, `rev_checkpoint_for` or `rev_checkpoint_while`; see [Checkpointing Schemes](schemes.md). Adding support for another rule system requires making the same two calls from its forward and reverse rules.
