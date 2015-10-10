module Knet
using Compat

# Print date, expression and elapsed time after execution
VERSION < v"0.4-" && eval(Expr(:using,:Dates))
macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end
export @date

include("util/gpu.jl");		export setseed
@useifgpu CUDArt
@useifgpu CUBLAS
@useifgpu CUSPARSE
@useifgpu CUDNN  
@gpu include("util/cudart.jl");
@gpu include("util/curand.jl");
@gpu include("util/cudnn.jl");
@gpu include("util/cusparse.jl");
include("util/linalg.jl");	

include("util/array.jl");	export isapprox
include("util/colops.jl");	export csize, clength, ccount, csub, cget, size2
include("data.jl");		export ItemTensor  # uses KUdense

include("op.jl");		
include("op/add.jl");		export add
include("op/dot.jl");		# export dot # this already has a definition in base
include("op/mul.jl");		export mul
include("op/input.jl");		export input
include("op/par.jl");           export par, Par, Gaussian, Uniform, Constant, Identity
include("op/loss.jl");		export quadloss, softloss, zeroone # TODO-TEST: logploss, xentloss, percloss, scalloss, 
include("op/actf.jl");		export sigm, tanh, relu, soft, logp
include("op/conv.jl");		# export conv # this already has a definition in base
include("op/pool.jl");		export pool
include("op/compound.jl");	export wdot, bias, wb, wbf, add2, lstm, irnn, wconv, convpool
include("update.jl");		

include("net.jl");              export Net, params, forw, back
include("net/initforw.jl")
include("net/initback.jl")
include("net/forw.jl")
include("net/back.jl")
include("net/util.jl")

include("model.jl");		export Model, train, test, predict, setopt!
include("model/fnn.jl");        export FNN
include("model/rnn.jl");        export RNN
include("model/s2c.jl");        export S2C

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

# include("util/deepcopy.jl");	# export cpucopy, gpucopy
# include("util/array.jl");	# export BaseArray

# # include("param.jl");		# export KUparam, setopt! # deprecated
# isdefined(:KUdense) || include("util/dense.jl");	# deprecate?
