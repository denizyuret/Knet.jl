import CUDA.CUDNN: cudnnTensorDescriptor, cudnnFilterDescriptor
using AutoGrad: Value

cudnnTensorDescriptor(x::Value) = cudnnTensorDescriptor(value(x))
cudnnFilterDescriptor(x::Value) = cudnnFilterDescriptor(value(x))
