# TODO: do better with systematically showing imports, exports per file and per module
module KnetArrays
import Knet, AutoGrad, ..Ops20
using ..Ops20: mat
using AutoGrad: @primitive, @primitive1, @zerograd, Node, Tape, Result, Value, Param, value
using Knet: @dbg

include("ops.jl")
include("gpu.jl"); export gpu, libknet8
include("kptr.jl"); export knetgc, gc, cuallocator
include("karray.jl"); export KnetArray
include("binary.jl")
include("bmm.jl")
include("conv.jl")
include("rnn.jl"); export RNN, rnnparam, rnnparams
include("cuarray.jl")
include("dropout.jl")
include("gcnode.jl"); export knetgcnode
include("linalg.jl")
include("reduction.jl")
include("statistics.jl")
include("unary.jl")
include("loss.jl")
include("train.jl")
include("batchnorm.jl"); export batchnorm, bnmoments, bnparams
include("serialize.jl"); export cpucopy, gpu, gpucopy
include("jld.jl"); export save, load, @save, @load
end
