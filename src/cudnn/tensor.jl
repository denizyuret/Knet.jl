import CUDA.CUDNN: cudnnTensorDescriptor, cudnnFilterDescriptor

cudnnTensorDescriptor(a::KnetArray; o...) = cudnnTensorDescriptor(CuArray(x); o...)

cudnnFilterDescriptor(a::KnetArray; o...) = cudnnFilterDescriptor(CuArray(x); o...)

