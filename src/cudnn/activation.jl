export activationForward
import Base: unsafe_convert
using Knet.KnetArrays: DevArray
using AutoGrad: AutoGrad, @primitive1

using CUDA.CUDNN: handle,
    cudnnActivationForward,
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
    cudnnNanPropagation_t,
        CUDNN_NOT_PROPAGATE_NAN,
        CUDNN_PROPAGATE_NAN


@primitive1 activationForward(x; o...),dy,y  activationBackward(x,y,dy; o...)
@primitive1 activationBackward(x,y...;o...)  throw(MethodError(back,activationBackward))

function activationForward(x::R; alpha=1, o...) where {T,R<:DevArray{T}}
    y, td = similar(x), TD(T,(1,1,length(x),1))
    cudnnActivationForward(handle(), AD(; o...), Ref(T(alpha)), td, x, Ref(T(0)), td, y)
    return y
end


function activationBackward(x::R,y::R,dy::R; alpha=1, o...) where {T,R<:DevArray{T}}
    dx, td = similar(x), TD(T,(1,1,length(x),1))
    cudnnActivationBackward(handle(), AD(; o...), Ref(T(alpha)), td, y, td, dy, td, x, Ref(T(0)), td, dx)
    return dx
end


"cudnnActivationDescriptor"
mutable struct _AD; ptr; end
unsafe_convert(::Type{<:Ptr}, ad::_AD)=ad.ptr

function AD(; # Defaults:
            mode::cudnnActivationMode_t=CUDNN_ACTIVATION_RELU,
            reluNanOpt::cudnnNanPropagation_t=CUDNN_NOT_PROPAGATE_NAN,
            coef::Float64=1.0
            )
    ptr = cudnnActivationDescriptor_t[C_NULL]
    cudnnCreateActivationDescriptor(ptr)
    cudnnSetActivationDescriptor(ptr[1], mode, reluNanOpt, coef)
    ad = _AD(ptr[1])
    finalizer(x->cudnnDestroyActivationDescriptor(x.ptr), ad)
    return ad
end
