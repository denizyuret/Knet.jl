module Ops21

import Knet.Ops20
using AutoGrad: AutoGrad, @primitive
using SpecialFunctions: erf

const dropout = Ops20.dropout; export dropout
const bmm = Ops20.bmm; export bmm
include("activation.jl"); export elu, gelu, relu, selu, sigm
include("mmul.jl"); export mmul

end
