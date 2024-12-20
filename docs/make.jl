using Revise
using XGMRES
using Documenter

DocMeta.setdocmeta!(XGMRES, :DocTestSetup, :(using XGMRES); recursive=true)

makedocs(;
    modules=[XGMRES],
    authors="bvieuble <bastien.vieuble@anss.ac.cn> and contributors",
    repo="https://github.com/bvieuble/XGMRES.jl/blob/{commit}{path}#{line}",
    sitename="XGMRES.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://bvieuble.github.io/XGMRES.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Overview" => "index.md",
        "XGMRES" => "xgmres.md",
        "Preconditioners" => "preconditioners.md",
        "Auxiliary Functions" => "auxiliary.md"
    ],
)

deploydocs(;
    repo="github.com/bvieuble/XGMRES.jl",
    devbranch="main",
)

