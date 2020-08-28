export dropoutForward
import Base: unsafe_convert
using Knet.KnetArrays: DevArray
using AutoGrad: AutoGrad, @primitive1
using CUDA: CuArray

using CUDA.CUDNN: handle,
    cudnnDropoutForward,
    cudnnDropoutBackward,
    cudnnDropoutGetStatesSize,
    cudnnDropoutGetReserveSpaceSize,
    cudnnDropoutDescriptor_t,
        cudnnCreateDropoutDescriptor,
        unsafe_cudnnSetDropoutDescriptor,
        cudnnGetDropoutDescriptor,
        cudnnRestoreDropoutDescriptor,
        cudnnDestroyDropoutDescriptor


mutable struct DropoutDescriptor; ptr; end
unsafe_convert(::Type{<:Ptr}, dd::DropoutDescriptor)=dd.ptr


# cudnnDropoutForward() doc says:
# "This function should not be running concurrently with another cudnnDropoutForward() function using the same states."
# So I am going to assume using a single buffer is fine until we have to deal with concurrency.
# TODO: fix this based on default_rng in Random
GLOBAL_DROPOUT_STATE = Ref{CuArray{Int,1}}()

# To debug gradients set GLOBAL_DROPOUT_SEED[] >= 0 for all dropout operations
GLOBAL_DROPOUT_SEED = Ref{Int}(-1)

# cudnnSetDropoutDescriptor is expensive, so let's cache the descriptors based on dropout probability
GLOBAL_DROPOUT_DESCRIPTORS = Dict{Cfloat,DropoutDescriptor}()


@primitive1 dropoutForward(x; dropout=0.5, reserveSpace=Ref{Any}(), o...),dy,y  dropoutBackward(dy; dropout=dropout, reserveSpace=reserveSpace)
@primitive1 dropoutBackward(dy;o...)  throw(MethodError(back,dropoutBackward))


function dropoutForward(x::R; dropout=0.5, reserveSpace=Ref{Any}()) where {T,R<:DevArray{T}}
    y, td, dd = similar(x), TD(T, (1,1,length(x),1)), DD(dropout)
    if GLOBAL_DROPOUT_SEED[] >= 0
        # This is a very expensive call (40x dropout), so only use for debugging
        @warn "Knet.CUDNN.GLOBAL_DROPOUT_SEED >= 0: calling expensive cudnnSetDropoutDescriptor" maxlog=1
        @cudnn_retry unsafe_cudnnSetDropoutDescriptor(dd, handle(), dropout, GLOBAL_DROPOUT_STATE[], sizeof(GLOBAL_DROPOUT_STATE[]), GLOBAL_DROPOUT_SEED[])
    end
    if !isassigned(reserveSpace)
        # reserveSpace is 1/8 of tensor size and is specific to this one call
        rss = Csize_t[0]; cudnnDropoutGetReserveSpaceSize(td, rss)
        reserveSpace[] = CuArray{Int}(undef, (rss[1]-1)÷sizeof(Int)+1)
    end
    cudnnDropoutForward(handle(), dd, td, x, td, y, reserveSpace[], sizeof(reserveSpace[]))
    return y
end


function dropoutBackward(dy::R; dropout, reserveSpace) where {T,R<:DevArray{T}}
    #@show summary.((dy,dropoutDesc[],reserveSpace[]))
    dx, td, dd = similar(dy), TD(T, (1,1,length(dy),1)), DD(dropout)
    cudnnDropoutBackward(handle(), dd, td, dy, td, dx, reserveSpace[], sizeof(reserveSpace[]))
    return dx
end


function DD(dropout::Real=0.5)
    get!(GLOBAL_DROPOUT_DESCRIPTORS, Cfloat(dropout)) do
        if !isassigned(GLOBAL_DROPOUT_STATE)
            ssize = Csize_t[0]; cudnnDropoutGetStatesSize(handle(), ssize)
            GLOBAL_DROPOUT_STATE[] = CuArray{Int}(undef, (ssize[1]-1)÷sizeof(Int)+1)
        end
        seed = floor(Culonglong,time())
        ptr = cudnnDropoutDescriptor_t[C_NULL]
        cudnnCreateDropoutDescriptor(ptr)
        @cudnn_retry unsafe_cudnnSetDropoutDescriptor(ptr[1], handle(), dropout, GLOBAL_DROPOUT_STATE[], sizeof(GLOBAL_DROPOUT_STATE[]), seed)
        dd = DropoutDescriptor(ptr[1])
        finalizer(x->cudnnDestroyDropoutDescriptor(x.ptr), dd)
        return dd
    end
end
