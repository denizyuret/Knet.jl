import Base: unsafe_convert
using Knet.KnetArrays: DevArray
using AutoGrad: AutoGrad, @primitive1

using CUDA.CUDNN: 
    #cudnnActivationForward,
    #cudnnActivationBackward,
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


mutable struct cudnnActivationDescriptor; ptr::cudnnActivationDescriptor_t; end

unsafe_convert(::Type{<:Ptr}, ad::cudnnActivationDescriptor)=ad.ptr

const cudnnActivationDescriptorCache = Dict{Tuple,cudnnActivationDescriptor}()

function cudnnActivationDescriptor(args...)
    get!(cudnnActivationDescriptorCache, args) do
        ptr = cudnnActivationDescriptor_t[C_NULL]
        cudnnCreateActivationDescriptor(ptr)
        cudnnSetActivationDescriptor(ptr[1], args...)
        ad = cudnnActivationDescriptor(ptr[1])
        finalizer(x->cudnnDestroyActivationDescriptor(x.ptr), ad)
        return ad
    end
end


function cudnnActivationForward(
    x::R, y::R = similar(x);
    mode::cudnnActivationMode_t = CUDNN_ACTIVATION_RELU,
    reluNanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    coef::Real=1,
    activationDesc::cudnnActivationDescriptor = cudnnActivationDescriptor(mode, reluNanOpt, Cdouble(coef)),
    alpha::Real=1,
    xDesc::cudnnTensorDescriptor = TD(x),
    beta::Real=0,
    yDesc::cudnnTensorDescriptor = xDesc
) where {T,R<:DevArray{T}}
    CUDA.CUDNN.cudnnActivationForward(handle(), activationDesc, Ref(T(alpha)), xDesc, x, Ref(T(beta)), yDesc, y)
    return y
end


function cudnnActivationBackward(
    x::R,y::R,dy::R,dx::R = similar(x);
    activationDesc::cudnnActivationDescriptor,
    alpha::Real,
    beta::Real,
    xDesc::cudnnTensorDescriptor,
    yDesc::cudnnTensorDescriptor = xDesc,
    dyDesc::cudnnTensorDescriptor = xDesc,
    dxDesc::cudnnTensorDescriptor = xDesc
) where {T,R<:DevArray{T}}
    CUDA.CUDNN.cudnnActivationBackward(handle(), activationDesc, Ref(T(alpha)), yDesc, y, dyDesc, dy, xDesc, x, Ref(T(beta)), dxDesc, dx)
    return dx
end


@primitive1(
    (cudnnActivationForward(
        x,y...;
        mode::cudnnActivationMode_t = CUDNN_ACTIVATION_RELU,
        reluNanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
        coef::Real=1,
        activationDesc::cudnnActivationDescriptor = cudnnActivationDescriptor(mode, reluNanOpt, coef),
        alpha::Real=1,
        xDesc::cudnnTensorDescriptor = TD(x),
        beta::Real=0,
        yDesc::cudnnTensorDescriptor = xDesc),
     _dy, _y),
    cudnnActivationBackward(
        x,_y,_dy;
        activationDesc = activationDesc,
        alpha = alpha,
        beta = beta,
        xDesc = xDesc))

@primitive1 cudnnActivationBackward(x,y...;o...)  throw(MethodError(back,cudnnActivationBackward))

