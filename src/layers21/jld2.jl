import JLD2
using CUDA.CUDNN: cudnnConvolutionDescriptor

JLD2.writeas(::Type{cudnnConvolutionDescriptor}) = Nothing
JLD2.wconvert(::Type{Nothing}, c::cudnnConvolutionDescriptor) = nothing
JLD2.rconvert(::Type{cudnnConvolutionDescriptor}, j::Nothing) = cudnnConvolutionDescriptor(C_NULL) # TODO: fix this
