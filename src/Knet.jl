VERSION >= v"0.4.0-dev+6521" && __precompile__()

module Knet
using Compat

# utilities for debugging and profiling.
macro dbg(i,x); if i & 0 != 0; esc(:(println(_dbg($x)))); end; end;

using AutoGrad; export grad, gradloss, gradcheck, getval

include("update.jl"); 		export Sgd, Momentum, Adam, Adagrad, Adadelta, Rmsprop, update!
include("distributions.jl"); 	export gaussian, xavier, bilinear
include("hyperopt.jl");         export hyperband, goldensection

"""
    Knet.dir(path...)

Construct a path relative to Knet root.

# Example
```julia
julia> Knet.dir("examples","mnist.jl")
"/home/dyuret/.julia/v0.5/Knet/examples/mnist.jl"
```
"""
dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)

end # module
