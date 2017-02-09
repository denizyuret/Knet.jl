using Documenter, Knet

makedocs(
    modules = [Knet],           # generate warnings for coverage
    clean = false,              # do we clean build/
    format = :html,
    sitename = "Knet.jl",
    authors = "Deniz Yuret and contributors.",
#    analytics = "UA-89508993-1",
#    linkcheck = !("skiplinks" in ARGS),
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        "Manual" => Any[
            "install.md",
            "README.md",
            "reference.md",
        ],
        "Textbook" => Any[
            "backprop.md",
            "softmax.md",
            "mlp.md",
            "cnn.md",
            "rnn.md",
            "rl.md",
            "opt.md",
            "gen.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/denizyuret/Knet.jl.git",
    target = "build",
    deps = nothing,
#   deps   = Deps.pip("mkdocs", "python-markdown-math"),
    make = nothing,
    julia = "0.5",
    osname = "linux",
)
