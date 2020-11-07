import CUDA.CUDNN: cudnnPoolingForwardAutoGrad
using CUDA.CUDNN: cudnnPoolingBackward, handle
using AutoGrad: AutoGrad, @primitive1, value

@primitive1((cudnnPoolingForwardAutoGrad(x; poolingDesc, alpha, beta, xDesc, yDesc, y),
             _dy,_y),
            ((x,y,dy,dx) = (value(x),value(_y),value(_dy),similar(x));
             cudnnPoolingBackward(handle(), poolingDesc, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx);
             dx))
