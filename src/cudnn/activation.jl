import CUDA.CUDNN: cudnnActivationForward, cudnnActivationForward!, cudnnActivationForwardAutoGrad
using CUDA.CUDNN: cudnnActivationBackward, handle
using AutoGrad: AutoGrad, @primitive1

cudnnActivationForward(x::KnetArray, d...; o...) = cudnnActivationForward!(similar(x), x, d...; o...)

cudnnActivationForward!(y::KnetArray, x::KnetArray, d...; o...) = (cudnnActivationForward!(CuArray(y), CuArray(x), d...; o...); y)

@primitive1(
    (cudnnActivationForwardAutoGrad(x; activationDesc, alpha, xDesc, beta, yDesc, y),
     _dy, _y),
    ((x, y, dy, dx) = (value(x), value(_y), value(_dy), similar(x));
     cudnnActivationBackward(handle(), activationDesc, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx);
     dx))
