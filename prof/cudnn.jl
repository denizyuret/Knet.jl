# Benchmarking cudnn activation functions; ours are faster so not included in Knet.jl

# typedef enum
# {
#     CUDNN_ACTIVATION_SIGMOID      = 0,
#     CUDNN_ACTIVATION_RELU         = 1,
#     CUDNN_ACTIVATION_TANH         = 2,
#     CUDNN_ACTIVATION_CLIPPED_RELU = 3,
#     CUDNN_ACTIVATION_ELU          = 4,
#     CUDNN_ACTIVATION_IDENTITY     = 5
# } cudnnActivationMode_t;

cudnnsigm(x)=cudnnActivationForward(x,mode=0)
cudnnrelu(x)=cudnnActivationForward(x,mode=1)
cudnntanh(x)=cudnnActivationForward(x,mode=2)
cudnncrelu(x;coef=1)=cudnnActivationForward(x,mode=3,coef=coef)
cudnnelu(x;coef=1)=cudnnActivationForward(x,mode=4,coef=coef)
cudnnidentity(x)=cudnnActivationForward(x,mode=5)

# typedef enum{
#     CUDNN_NOT_PROPAGATE_NAN  = 0,
#     CUDNN_PROPAGATE_NAN      = 1,
# } cudnnNanPropagation_t;

mutable struct AD; ptr; end
Base.unsafe_convert(::Type{Cptr}, ad::AD)=ad.ptr
function AD(; mode=0, coef=0, reluNanOpt=false)
    d = Cptr[0]
    @cudnn(cudnnCreateActivationDescriptor,(Ptr{Cptr},),d)
    @cudnn(cudnnSetActivationDescriptor,(Cptr,Cint,Cint,Cdouble),d[1],mode,(reluNanOpt ? 1 : 0), coef)
    ad = AD(d[1])
    finalizer(x->@cudnn(cudnnDestroyActivationDescriptor,(Cptr,),x.ptr), ad)
    return ad
end

t4d(x::KnetArray{T,1}) where T = TD(T,(1,1,length(x),1))
t4d(x::KnetArray{T,2}) where T = TD(T,(1,1,size(x,1),size(x,2)))
t4d(x::KnetArray{T,3}) where T = TD(T,(1,size(x)...))
t4d(x::KnetArray{T,4}) where T = TD(T,size(x))
t4d(x::KnetArray{T,5}) where T = TD(T,size(x))
t4d(x::KnetArray{T,N}) where {T,N} = TD(T,(1,1,length(x),1)) # TODO: do something smarter here

function cudnnActivationForward(x::KnetArray{T}; handle=cudnnhandle(), alpha=1, o...) where T # TODO: do not rely on KnetArray
    beta=0
    ad = AD(;o...)
    td = t4d(x)
    y = similar(x)
    @cudnn(cudnnActivationForward,(Cptr,Cptr,Ptr{T},Cptr,Ptr{T},Ptr{T},Cptr,Ptr{T}), handle, ad, Ref(T(alpha)), td, x, Ref(T(beta)), td, y) # TODO: understand Ref
    return y
end

function cudnnActivationBackward(dy::KnetArray{T},y::KnetArray{T},x::KnetArray{T}; handle=cudnnhandle(), alpha=1, o...) where T
    beta = 0
    ad = AD(;o...)
    td = t4d(x)
    dx = similar(x)
    @cudnn(cudnnActivationBackward,
          (Cptr,Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Ptr{T},Cptr,Ptr{T}),
          handle,ad, Ref(T(alpha)),td,y,td,dy,td,x,Ref(T(beta)),td,dx)
    return dx
end

@primitive cudnnActivationForward(x;o...),dy,y  cudnnActivationBackward(value(dy),value(y),value(x);o...)

