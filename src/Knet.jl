VERSION >= v"0.4.0-dev+6521" && __precompile__()

module Knet
using Compat

# utilities for debugging and profiling.
macro dbg(i,x); if i & 0 != 0; esc(:(println(_dbg($x)))); end; end;
macro gs(); if false; esc(:(ccall(("cudaDeviceSynchronize","libcudart"),UInt32,()))); end; end

const libknet8 = Libdl.find_library(["libknet8.so"], [dirname(@__FILE__)])

using AutoGrad; export grad, gradloss, gradcheck, getval

include("compat.jl");           # julia6 compat fixes
include("gpu.jl");              export gpu
include("gpuarrays.jl");        export mat, logsumexp
#include("kptr.jl");             # KnetPtr
#include("karray.jl");           export KnetArray
include("unfuse.jl");           # julia6 broadcast fixes
include("unary.jl");            export relu, sigm, invx, logp, dropout
broadcast_ops = [
("add",".+","xi+yi"),
("sub",".-","xi-yi"),
("mul",".*","xi*yi"),
("div","./","xi/yi"),
("pow",".^","pow(xi,yi)"),
("max","max","(xi>yi?xi:yi)"),
("min","min","(xi<yi?xi:yi)"),
("eq",".==","xi==yi"),
("ne",".!=","xi!=yi"),
("gt",".>","xi>yi"),
("ge",".>=","xi>=yi"),
("lt",".<","xi<yi"),
("le",".<=","xi<=yi"),
# "hypot",
# "rhypot",
# "atan2",
# "frexp",
# "ldexp",
# "scalbn",
# "scalbln",
# "jn",
# "yn",
# "fmod",
# "remainder",
# "mod",
# "fdim",
("invxback","invxback","(-xi*yi*yi)"),
("reluback","reluback","(yi>0?xi:0)"),
("sigmback","sigmback","(xi*yi*(1-yi))"),
("tanhback","tanhback","(xi*(1-yi*yi))"),
("rpow","rpow","pow(yi,xi)"),   # need this for Array.^Scalar
]

reduction_ops = [
("sum","sum","ai+xi","xi","0"),
("prod","prod","ai*xi","xi","1"),
("maximum","maximum","(ai>xi?ai:xi)","xi","(-INFINITY)"),
("minimum","minimum","(ai<xi?ai:xi)","xi","INFINITY"),
("sumabs","sumabs","ai+xi","(xi<0?-xi:xi)","0"),
("sumabs2","sumabs2","ai+xi","(xi*xi)","0"),
("maxabs","maxabs","(ai>xi?ai:xi)","(xi<0?-xi:xi)","0"),
("minabs","minabs","(ai<xi?ai:xi)","(xi<0?-xi:xi)","INFINITY"),
("countnz","countnz","ai+xi","(xi!=0)","0"),
]
export logsumexp
#include("linalg.jl");           export mat # matmul, axpy!, transpose, (i)permutedims
#include("conv.jl");             export conv4, pool, deconv4, unpool
include("update.jl"); 		export Sgd, Momentum, Adam, Adagrad, Adadelta, Rmsprop, update!
include("distributions.jl"); 	export gaussian, xavier, bilinear
include("random.jl");           export setseed
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


# See if we have a gpu at initialization:
function __init__()
    try
        r = gpu(true)
        # info(r >= 0 ? "Knet using GPU $r" : "No GPU found, Knet using the CPU")
    catch e
        gpu(false)
        # warn("Knet using the CPU: $e")
    end
end

end # module
