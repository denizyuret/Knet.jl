using Knet
include(gpu() ? "gputests.jl" : "cputests.jl")
