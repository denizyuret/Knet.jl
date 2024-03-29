using Documenter, Knet

# Avoid loading examples and their requirements
# Extract their docstrings instead
# Unfortunately this leads to wrong "source" links :(
# run(`make exampledocs`)
# include("exampledocs.jl")

# Load examples
# load_only = true
# for ex in ("linreg","housing","mnist","lenet","charlm","optimizers","vgg","resnet")
#     println("$ex.jl")
#     include(Knet.dir("examples","$ex.jl"))
# end
# println("Examples loaded")

makedocs(
    # Including modules prevents getting docstrings from Main 
    # Including Main in the list leads to too many warnings
    # modules = [Knet,AutoGrad,LinReg,Housing,MNIST,LeNet,CharLM,Optimizers,VGG,ResNet],
    modules = [Knet,AutoGrad],
    format = Documenter.HTML(),
    clean = false,              # do we clean build dir
    sitename = "Knet.jl",
    authors = "Deniz Yuret and contributors.",
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        "Manual" => Any[
            "install.md",
            "tutorial.md",
#           "examples.md",
            "reference.md",
        ],
        "Textbook Draft" => Any[
            "backprop.md",
            "softmax.md",
            "mlp.md",
            "cnn.md",
            "rnn.md",
            "rl.md",
            "opt.md",
            "gen.md",
            "nce.md",
            "vae.md",
        ],
    ],
#    doctest = true,
#    analytics = "UA-89508993-1",
#    linkcheck = !("skiplinks" in ARGS),
)

deploydocs(
    repo = "github.com/denizyuret/Knet.jl.git",
    # target = "build",
    # julia = "1.0",
    # osname = "linux",
    # make = nothing,
    # deps = nothing,
    # deps   = Deps.pip("mkdocs", "python-markdown-math"),
)
