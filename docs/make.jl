using Pkg

checkpointingspec = PackageSpec(path = joinpath(dirname(@__FILE__), ".."))
Pkg.develop(checkpointingspec)

# when first running instantiate
Pkg.instantiate()

using Documenter
using Checkpointing

makedocs(
    sitename = "Checkpointing.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX(),
    ),
    modules = [Checkpointing],
    repo = "https://github.com/Argonne-National-Laboratory/Checkpointing.jl/blob/{commit}{path}#{line}",
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "Schemes" => "schemes.md",
        "Rules" => "rules.md",
        "Storage" => "storage.md",
        "API" => "lib/checkpointing.md",
    ],
)

deploydocs(
    repo = "github.com/Argonne-National-Laboratory/Checkpointing.jl.git",
    target = "build",
    devbranch = "main",
    devurl = "main",
    push_preview = true,
)
