import CUDA.CUDNN: cudnnActivationForwardAutoGrad
using CUDA.CUDNN: cudnnActivationBackward, handle
using AutoGrad: AutoGrad, @primitive1, value

@primitive1(
    (cudnnActivationForwardAutoGrad(x; activationDesc, alpha, xDesc, beta, yDesc, y),
     _dy, _y),
    ((x, y, dy, dx) = (value(x), value(_y), value(_dy), similar(x));
     cudnnActivationBackward(handle(), activationDesc, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx);
     dx))
