# Uncomment this when all CUDA modules support precompilation
# isdefined(Base, :__precompile__) && __precompile__()
module Knet
using Compat

### GPU detection and initialization
include("util/gpu.jl");		export gpu, @gpu, @useifgpu, setseed
@useifgpu CUDArt
@useifgpu CUBLAS
@useifgpu CUSPARSE
@useifgpu CUDNN  

### GPU dependent includes
@gpu include("util/cudart.jl");		# export copysync!, fillsync!
@gpu include("util/curand.jl");
@gpu include("util/cusparse.jl");
@gpu include("util/deepcopy.jl"); 	# export cpucopy, gpucopy
@gpu include("util/linalg.jl");	

### GPU independent utilities
include("util/dbg.jl");		export @date, @dbg, gpusync
include("util/rgen.jl");	export Gaussian, Uniform, Constant, Identity, Xavier, Bernoulli
include("util/array.jl");	# export isapprox
include("util/colops.jl");	export csize, clength, ccount, csub, cget, size2, minibatch

### Main
include("op.jl")
include("op/actf.jl")
include("op/broadcast.jl")
include("op/conv.jl")
include("op/dot.jl")
include("op/lrn.jl")
include("op/nce.jl")
include("op/pool.jl")
include("op/genop.jl")
include("net.jl");              export setp, wnorm, gnorm, reset!, clean #export Reg, Net, set!, inc!, registers, params, ninputs, out, dif, stack_isempty, stack_empty!, 
include("net/forw.jl");         export forw, sforw
include("net/back.jl");         export back, sback
include("net/initforw.jl")
include("net/initback.jl")
include("compiler.jl");		export @knet, compile, _comp_parse_def # @knet needs _comp_parse_def
include("gradcheck.jl");  	export gradcheck 
include("update.jl");		export update!
include("loss.jl");		export quadloss, softloss, zeroone # TODO-TEST: logploss, xentloss, percloss, scalloss, 
include("kfun.jl")

# To be deprecated:
include("data/ItemTensor.jl");		export ItemTensor
include("data/S2SData.jl");     	export S2SData, maxtoken
include("data/SequencePerLine.jl"); 	export SequencePerLine
include("data/SketchEngine.jl"); 	export SketchEngine
include("data/TagData.jl"); 		export TagData, sponehot

# Load kernels from CUDArt
function __init__()
# Let's just use one gpu for now.
#    @gpu CUDArt.init(CUDArt.devices(i->true))
    @gpu CUDArt.init(CUDArt.device())
end

end # module
