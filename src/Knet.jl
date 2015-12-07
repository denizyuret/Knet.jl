"Module that contains the names of Knet functions."
module Kenv; kdef(x,y)=eval(Kenv,Expr(:(=),x,Expr(:quote,y))); end

module Knet
using Compat
using Main.Kenv
#using Kenv.kdef # This does not work for some reason, have to use Kenv.kdef explicitly

# This is for debugging
#DBG=false; dbg()=DBG; dbg(b::Bool)=(global DBG=b)
#macro dbg(x) :(DBG && $(esc(x))) end
# This is for production
macro dbg(x) nothing end        

#gpusync()=device_synchronize() # This is for profiling
gpusync()=nothing               # This is for production

include("util/gpu.jl");		# Find out if we have a gpu, defines gpu(), @gpu, @useifgpu etc.

# Useful utilities
macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end # Print date, expression; run and print elapsed time after execution

setseed(n)=srand(n)             # This gets overwritten in curand.jl if gpu available

export @date, @dbg, gpusync, setseed

@useifgpu CUDArt
@useifgpu CUBLAS
@useifgpu CUSPARSE
@useifgpu CUDNN  
@gpu include("util/cudart.jl");
@gpu include("util/curand.jl");
@gpu include("util/cusparse.jl");
@gpu include("util/deepcopy.jl");	export cpucopy, gpucopy
include("util/linalg.jl");	
include("util/rgen.jl");	export Gaussian, Uniform, Constant, Identity, Xavier, Bernoulli

include("util/array.jl");	export isapprox
include("util/colops.jl");	export csize, clength, ccount, csub, cget, size2

include("op.jl");		
include("op/actf.jl")
include("op/add.jl")
include("op/arr.jl")
include("op/conv.jl")
include("op/dot.jl")
include("op/input.jl")
include("op/mul.jl")
include("op/nce.jl")
include("op/par.jl")
include("op/pool.jl")
include("op/rnd.jl")

include("update.jl");		export update!
include("op/loss.jl");		export quadloss, softloss, zeroone # TODO-TEST: logploss, xentloss, percloss, scalloss, 

include("model.jl");		export Model, train, test, predict, setopt!, wnorm, gnorm
include("net.jl");              export Reg, Net, set!, inc!, registers, params, ninputs, out, dif
include("compiler.jl");		export @knet, compile, _comp_parse_def # @knet needs the last one
include("net/initforw.jl")
include("net/initback.jl")
include("net/forw.jl");         export forw, forwtest
include("net/back.jl");         export back
include("net/util.jl");         export reset!

include("op/compound.jl");	# export wdot, bias, wb, wf, wbf, add2, lstm, irnn, wconv, cbfp # repeat,drop in base

include("model/gradcheck.jl");  export gradcheck
include("model/fnn.jl");        export FNN
include("model/rnn.jl");        export RNN
include("model/s2c.jl");        export S2C
include("model/s2s.jl");        export S2S, S2SData, encoder, decoder # last two needed by the compiler
include("model/tagger.jl");	export Tagger
include("model/nce.jl");	export NCE

include("data/ItemTensor.jl");		export ItemTensor
include("data/S2SData.jl");     	export S2SData, maxtoken
include("data/SequencePerLine.jl"); 	export SequencePerLine
include("data/SketchEngine.jl"); 	export SketchEngine
include("data/TagData.jl"); 		export TagData, sponehot

end # module

# include("op/mmul.jl");     	# export Mmul
# include("op/bias.jl");		# export Bias
# include("op/drop.jl");		# export Drop
# include("op/add2.jl");		# export Add2
# include("op/mul2.jl");		# export Mul2
# include("op/actf.jl");		# export Actf, Logp, Relu, Sigm, Soft, Tanh
# include("op/loss.jl");		# export Loss, QuadLoss, SoftLoss, LogpLoss, XentLoss, PercLoss, ScalLoss

# include("netcomp.jl");		# export Net
# include("compiler.jl")
# include("net.jl");		# export Net
# include("mlp.jl");		# export MLP, predict # , accuracy, loadnet, savenet

# include("model/irnn.jl");	# export IRNN
# include("model/lstm.jl");	# export LSTM
# include("model/s2c.jl");	# export S2C
# include("model/kperceptron.jl"); # export KPerceptron

# include("data/adding.jl");	# export Adding
# include("data/mnist.jl");	# export MNIST
# include("data/pixels.jl");	# export Pixels

# include("util/array.jl");	# export BaseArray

# # include("param.jl");		# export KUparam, setopt! # deprecated
# isdefined(:KUdense) || include("util/dense.jl");	# deprecate?

# _KENV is the environment table for knet functions.  We keep these
# out of Julia to avoid name conflicts.
# isdefined(:_KENV) || (_KENV = Dict{Symbol,Any}())

