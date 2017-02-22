using Documenter, Knet

# Avoid loading examples and their requirements
# Extract their docstrings instead
# Unfortunately this leads to wrong "source" links :(
# run(`make exampledocs`)
# include("exampledocs.jl")

# Load examples
load_only = true
for ex in ("linreg","housing","mnist","lenet","charlm","optimizers","vgg","resnet","hyperband")
    println("$ex.jl")
    include(Knet.dir("examples","$ex.jl"))
end
println("Examples loaded")

makedocs(
    # Including modules prevents getting docstrings from Main 
    # Including Main in the list leads to too many warnings
    # modules = [Knet,AutoGrad,LinReg,Housing,MNIST,LeNet,CharLM,Optimizers,VGG,ResNet],
    clean = false,              # do we clean build dir
    format = :html,
    sitename = "Knet.jl",
    authors = "Deniz Yuret and contributors.",
#    analytics = "UA-89508993-1",
#    linkcheck = !("skiplinks" in ARGS),
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        "Manual" => Any[
            "install.md",
            "tutorial.md",
            "examples.md",
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
#   deps = nothing,
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    make = nothing,
    julia = "0.5",
    osname = "linux",
)
