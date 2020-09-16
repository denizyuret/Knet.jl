using AutoGrad: AutoGrad, @primitive1

using CUDA.CUDNN: 
    #cudnnActivationForward,
    cudnnActivationBackward,
    cudnnActivationDescriptor_t,
        cudnnCreateActivationDescriptor,
        cudnnSetActivationDescriptor,
        cudnnGetActivationDescriptor,
        cudnnDestroyActivationDescriptor,
    cudnnActivationMode_t,
        CUDNN_ACTIVATION_SIGMOID,
        CUDNN_ACTIVATION_RELU,
        CUDNN_ACTIVATION_TANH,
        CUDNN_ACTIVATION_CLIPPED_RELU,
        CUDNN_ACTIVATION_ELU,
        CUDNN_ACTIVATION_IDENTITY,
    handle

function cudnnActivationForward(
    x, y = similar(x);
    mode::cudnnActivationMode_t = CUDNN_ACTIVATION_RELU,
    reluNanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    coef::Real=1,
    activationDesc::cudnnActivationDescriptor = cudnnActivationDescriptor(mode, reluNanOpt, Cdouble(coef)),
    alpha::Real=1,
    xDesc::cudnnTensorDescriptor = TD(x),
    beta::Real=0,
    yDesc::cudnnTensorDescriptor = xDesc)
    alpha, beta = scalr(alpha,x), scalr(beta,x)
    _cudnnActivationForward(x; activationDesc, alpha, xDesc, beta, yDesc, y)
end

function _cudnnActivationForward(x; activationDesc, alpha, xDesc, beta, yDesc, y)
    CUDA.CUDNN.cudnnActivationForward(handle(), activationDesc, alpha, xDesc, x, beta, yDesc, y)
    return y
end

@primitive1(
    (_cudnnActivationForward(x; activationDesc, alpha, xDesc, beta, yDesc, y),
     _dy, _y),
    ((x, y, dy, dx) = (value(x), value(_y), value(_dy), similar(x));
     cudnnActivationBackward(handle(), activationDesc, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx);
     dx))
