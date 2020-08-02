# TODO: do better with systematically showing imports, exports per file and per module
module KnetArrays
import Knet, AutoGrad, ..Ops20
using ..Ops20: mat
using AutoGrad: @primitive, @primitive1, @zerograd, Node, Tape, Result, Value, Param, value

# To see debug output, start julia with `JULIA_DEBUG=Knet julia`
# To perform profiling, set ENV["KNET_TIMER"] to "true" and rebuild Knet. (moved this to gpu.jl)
# The @dbg macro below evaluates `ex` only when debugging. The @debug macro prints stuff as documented in Julia.
macro dbg(ex); :(if Base.CoreLogging.current_logger_for_env(Base.CoreLogging.Debug,:none,Knet)!==nothing; $(esc(ex)); end); end

include("ops.jl")
include("gpu.jl"); export gpu, libknet8
include("kptr.jl"); export knetgc, gc, cuallocator
include("karray.jl"); export KnetArray
include("binary.jl")
include("bmm.jl")
include("conv.jl")
include("rnn.jl"); export RNN, rnnparam, rnnparams, rnninit, rnnforw
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
