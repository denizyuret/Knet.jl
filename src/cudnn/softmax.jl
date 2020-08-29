using Knet.KnetArrays: DevArray
using AutoGrad: AutoGrad, @primitive1

import CUDA.CUDNN:
    cudnnSoftmaxForward,
    cudnnSoftmaxBackward
using CUDA.CUDNN:
    cudnnSoftmaxAlgorithm_t,
        CUDNN_SOFTMAX_FAST,     # 0, /* straightforward implementation */
        CUDNN_SOFTMAX_ACCURATE, # 1, /* subtract max from every point to avoid overflow */
        CUDNN_SOFTMAX_LOG,      # 2
    cudnnSoftmaxMode_t,
        CUDNN_SOFTMAX_MODE_INSTANCE, # 0, /* compute the softmax over all C, H, W for each N */
        CUDNN_SOFTMAX_MODE_CHANNEL,  # 1  /* compute the softmax over all C for each H, W, N */
    handle


function cudnnSoftmaxForward(x::R; 
                             algo::cudnnSoftmaxAlgorithm_t = CUDNN_SOFTMAX_FAST,
                             mode::cudnnSoftmaxMode_t = CUDNN_SOFTMAX_MODE_INSTANCE,
                             alpha::Real = 1,
                             xDesc::cudnnTensorDescriptor = TD(x),
                             beta::Real = 0,
                             yDesc::cudnnTensorDescriptor = xDesc,
                             y::R = similar(x)
                             ) where {T,R<:DevArray{T}}
    cudnnSoftmaxForward(handle(), algo, mode, Ref(T(alpha)), xDesc, x, Ref(T(beta)), yDesc, y)
    return y
end


function cudnnSoftmaxBackward(y::R, dy::R;
                              algo::cudnnSoftmaxAlgorithm_t,
                              mode::cudnnSoftmaxMode_t,
                              alpha::Real,
                              yDesc::cudnnTensorDescriptor,
                              dyDesc::cudnnTensorDescriptor,
                              beta::Real,
                              dxDesc::cudnnTensorDescriptor = yDesc,
                              dx::R = similar(y)
                              ) where {T,R<:DevArray{T}}
    cudnnSoftmaxBackward(handle(), algo, mode, Ref(T(alpha)), yDesc, y, dyDesc, dy, Ref(T(beta)), dxDesc, dx)
    return dx
end


@primitive1((cudnnSoftmaxForward(x; 
                                 algo::cudnnSoftmaxAlgorithm_t = CUDNN_SOFTMAX_FAST,
                                 mode::cudnnSoftmaxMode_t = CUDNN_SOFTMAX_MODE_INSTANCE,
                                 alpha::Real = 1,
                                 xDesc::cudnnTensorDescriptor = TD(x),
                                 beta::Real = 0,
                                 yDesc::cudnnTensorDescriptor = xDesc,
                                 y = similar(x)),_dy,_y),
            cudnnSoftmaxBackward(_y, _dy;
                                 algo = algo,
                                 mode = mode,
                                 alpha = alpha,
                                 yDesc = xDesc,
                                 dyDesc = xDesc,
                                 beta = beta))

@primitive1 cudnnSoftmaxBackward(x,y...;o...)  throw(MethodError(back,cudnnSoftmaxBackward))
