push!(LOAD_PATH, "../src/")

using Documenter, FiniteDiffWENO5

makedocs(
    sitename = "FiniteDiffWENO5",
    authors = "Hugo Dominguez",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "GettingStarted.md",
        "Background" => "background.md",
        "GPU and backends" => "GPU.md",
        "API" => "API.md",
    ],
)

deploydocs(
    repo = "github.com/JuliaGeodynamics/FiniteDiffWENO5.jl.git",
    branch = "gh-pages",
    target = "build",
    devbranch = "main",
    devurl = "dev",
    forcepush = true,
    push_preview = false,
)
