import CUDA.CUDNN: cudnnSoftmaxForwardAutoGrad
using CUDA.CUDNN: cudnnSoftmaxBackward, handle
using AutoGrad: AutoGrad, @primitive1, value

@primitive1((cudnnSoftmaxForwardAutoGrad(x; algo, mode, alpha, xDesc, beta, yDesc, y),
             _dy,_y),
            ((x,y,dy,dx) = (value(x),value(_y),value(_dy),similar(x));
             cudnnSoftmaxBackward(handle(), algo, mode, alpha, yDesc, y, yDesc, dy, beta, xDesc, dx);
             dx))
