import CUDA.CUDNN: cudnnDropoutForwardAD
using CUDA.CUDNN: cudnnDropoutBackward, handle
using AutoGrad: AutoGrad, @primitive1, value

@primitive1((cudnnDropoutForwardAD(x; xDesc, y, yDesc, dropoutDesc, reserveSpace),
             _dy, _y),
            ((x,y,dy,dx) = (value(x),value(_y),value(_dy),similar(x));
             cudnnDropoutBackward(handle(), dropoutDesc, yDesc, dy, xDesc, dx, reserveSpace, sizeof(reserveSpace));
             dx))
