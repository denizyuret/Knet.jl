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
        CUDNN_ACTIVATION_SIGMOID,      # 0
        CUDNN_ACTIVATION_RELU,         # 1
        CUDNN_ACTIVATION_TANH,         # 2
        CUDNN_ACTIVATION_CLIPPED_RELU, # 3
        CUDNN_ACTIVATION_ELU,          # 4
        CUDNN_ACTIVATION_IDENTITY,     # 5
    cudnnNanPropagation_t,
        CUDNN_NOT_PROPAGATE_NAN, # 0
        CUDNN_PROPAGATE_NAN,     # 1
    handle


cudnnActivationForward(x; o...)                     = cudnnActivationForwardWithDefaults(x; o...)
cudnnActivationForward(x, activationDesc; o...)     = cudnnActivationForwardWithDefaults(x; activationDesc, o...)
cudnnActivationForward!(y, x; o...)                 = cudnnActivationForwardWithDefaults(x; y, o...)
cudnnActivationForward!(y, x, activationDesc; o...) = cudnnActivationForwardWithDefaults(x; y, activationDesc, o...)


# This non-public function is used so that we can declare default values for kwargs only once.
function cudnnActivationForwardWithDefaults(
    x;
    y = similar(x),
    mode::cudnnActivationMode_t = CUDNN_ACTIVATION_RELU,
    reluNanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    coef::Real=1,
    activationDesc::cudnnActivationDescriptor = cudnnActivationDescriptor(mode, reluNanOpt, Cdouble(coef)),
    alpha::Real=1,
    beta::Real=0,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x),
    yDesc::cudnnTensorDescriptor = xDesc
)
    alpha, beta = scalr(alpha,x), scalr(beta,x)
    cudnnActivationForwardAutoGrad(x; activationDesc, alpha, xDesc, beta, yDesc, y)
end


# This non-public function is used to define gradients.
function cudnnActivationForwardAutoGrad(x; activationDesc, alpha, xDesc, beta, yDesc, y)
    CUDA.CUDNN.cudnnActivationForward(handle(), activationDesc, alpha, xDesc, x, beta, yDesc, y)
    return y
end


# Define gradients
@primitive1(
    (cudnnActivationForwardAutoGrad(x; activationDesc, alpha, xDesc, beta, yDesc, y),
     _dy, _y),
    ((x, y, dy, dx) = (value(x), value(_y), value(_dy), similar(x));
     cudnnActivationBackward(handle(), activationDesc, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx);
     dx))
