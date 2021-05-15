module Ops21

using AutoGrad: AutoGrad, @primitive
using SpecialFunctions: erf

include("activation.jl"); export elu, gelu, hardsigmoid, hardswish, relu, selu, sigm, swish
include("linear.jl"); export linear
include("batchnorm.jl"); export batchnorm
include("conv.jl"); export conv
include("pool.jl"); export pool, mean
include("zeropad.jl"); export zeropad
include("reshape2d.jl"); export reshape2d

import Knet.Ops20
const dropout = Ops20.dropout; export dropout
const bmm = Ops20.bmm; export bmm

end
