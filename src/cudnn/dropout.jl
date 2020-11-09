import CUDA.CUDNN: cudnnDropoutForwardAutoGrad
using CUDA.CUDNN: cudnnDropoutBackward
using AutoGrad: AutoGrad, @primitive1, value

@primitive1((cudnnDropoutForwardAutoGrad(x; xDesc, y, yDesc, dropout, dropoutDesc, reserveSpace),
             _dy, _y),
            ((x,y,dy,dx) = (value(x),value(_y),value(_dy),similar(x));
             cudnnDropoutBackward(handle(), dropoutDesc, yDesc, dy, xDesc, dx, reserveSpace, sizeof(reserveSpace));
             dx))
