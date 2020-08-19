module Ops20_gpu

include("cudnn_retry.jl")
include("activation.jl")
include("batchnorm.jl")
include("bmm.jl")
include("conv.jl")
include("dropout.jl")
include("softmax.jl")
include("loss.jl")
include("rnn.jl")

end
