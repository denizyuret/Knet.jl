module Ops21

using Knet.Ops20: elu, relu, selu, sigm
using AutoGrad: AutoGrad, @primitive

include("activation.jl"); export elu, gelu, relu, selu, sigm

end
