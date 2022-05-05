using Pkg

diffractorspec = PackageSpec(url="https://github.com/JuliaDiff/Diffractor.jl", rev="main")
Pkg.add([diffractorspec])
checkpointingspec = PackageSpec(path=joinpath(dirname(@__FILE__), ".."))
Pkg.develop(checkpointingspec)

# when first running instantiate
Pkg.instantiate()

using Documenter
using Checkpointing

makedocs(
    sitename = "Checkpointing.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX()
    ),
    modules = [Checkpointing],
    repo = "https://github.com/Argonne-National-Laboratory/Checkpointing.jl/blob/{commit}{path}#{line}",
    strict = true,
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "Library" => [
        "Checkpointing" => "lib/checkpointing.md",
        ],
    ]
)

deploydocs(
    repo = "github.com/Argonne-National-Laboratory/Checkpointing.jl.git",
    target = "build",
    devbranch = "main",
    devurl = "main",
    push_preview = true,
)