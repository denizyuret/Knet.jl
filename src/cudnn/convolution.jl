import CUDA.CUDNN: cudnnConvolutionForwardAutoGrad
using CUDA.CUDNN: cudnnConvolutionBwdFilterAlgoPerf, @workspace, cudnnConvolutionBackwardFilter, cudnnConvolutionBwdDataAlgoPerf, cudnnConvolutionBackwardData, cudnnConvolutionBackwardBias, scalingParameter
using AutoGrad: AutoGrad, @primitive1, value
using Knet.Ops20: reluback

# Define gradients
@primitive1(
    (cudnnConvolutionForwardAutoGrad(w, x, bias, z; y, activation, convDesc, wDesc, xDesc, yDesc, zDesc, biasDesc, alpha, beta, dy),
     _dy,_y),

    (dw = similar(w);
     dy[] = (activation==CUDNN_ACTIVATION_IDENTITY ? value(_dy) : reluback.(value(_dy),value(_y)));
     x = value(x);
     beta = scalingParameter(eltype(x),0);
     p = cudnnConvolutionBwdFilterAlgoPerf(xDesc, x, yDesc, dy[], convDesc, wDesc, dw);
     @workspace size=p.memory workspace->cudnnConvolutionBackwardFilter(handle(), alpha, xDesc, x, yDesc, dy[], convDesc, p.algo, workspace, sizeof(workspace), beta, wDesc, dw);
     dw),

    ((w,dx) = (value(w),similar(x));
     beta = scalingParameter(eltype(x),0);
     p = cudnnConvolutionBwdDataAlgoPerf(wDesc, w, yDesc, dy[], convDesc, xDesc, dx);
     @workspace size=p.memory workspace->cudnnConvolutionBackwardData(handle(), alpha, wDesc, w, yDesc, dy[], convDesc, p.algo, workspace, sizeof(workspace), beta, xDesc, dx);
     dx),

    (db = similar(bias);
     T=eltype(db); (alpha,beta) = (scalingParameter(T,1), scalingParameter(T,0));
     cudnnConvolutionBackwardBias(handle(), alpha, yDesc, dy[], beta, biasDesc, db);
     db),

    (beta[] * dy[]))
