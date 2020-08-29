import Base: unsafe_convert
using Knet.KnetArrays: DevArray
using AutoGrad: AutoGrad, @primitive1
using CUDA: CuArray

import CUDA.CUDNN: 
    cudnnDropoutForward,
    cudnnDropoutBackward
using CUDA.CUDNN: 
    cudnnDropoutDescriptor_t,
        cudnnCreateDropoutDescriptor,
        cudnnSetDropoutDescriptor,
        cudnnGetDropoutDescriptor,
        cudnnRestoreDropoutDescriptor,
        cudnnDestroyDropoutDescriptor,
    cudnnDropoutGetStatesSize,
    cudnnDropoutGetReserveSpaceSize,
    handle


# cudnnDropoutForward() doc says:
# "This function should not be running concurrently with another cudnnDropoutForward() function using the same states."
# So I am going to assume using a single buffer is fine until we have to deal with concurrency.
# TODO: fix this based on default_rng in Random
cudnnDropoutState = Ref{CuArray{Int,1}}()

# To debug gradients set cudnnDropoutSeed[] >= 0 which makes all dropout operations deterministic
cudnnDropoutSeed = Ref{Int}(-1)


mutable struct cudnnDropoutDescriptor; ptr::cudnnDropoutDescriptor_t; end
unsafe_convert(::Type{<:Ptr}, dd::cudnnDropoutDescriptor)=dd.ptr
const cudnnDropoutDescriptorCache = Dict{Cfloat,cudnnDropoutDescriptor}()
function cudnnDropoutDescriptor(dropout::Real)
    get!(cudnnDropoutDescriptorCache, Cfloat(dropout)) do
        if !isassigned(cudnnDropoutState)
            ssize = Csize_t[0]; cudnnDropoutGetStatesSize(handle(), ssize)
            cudnnDropoutState[] = CuArray{Int}(undef, (ssize[1]-1)÷sizeof(Int)+1)
        end
        seed = floor(Culonglong,time())
        ptr = cudnnDropoutDescriptor_t[C_NULL]
        cudnnCreateDropoutDescriptor(ptr)
        @retry cudnnSetDropoutDescriptor(ptr[1], handle(), dropout, cudnnDropoutState[], sizeof(cudnnDropoutState[]), seed)
        dd = cudnnDropoutDescriptor(ptr[1])
        finalizer(x->cudnnDestroyDropoutDescriptor(x.ptr), dd)
        return dd
    end
end


function cudnnDropoutForward(x::R;
                             dropout::Real = 0.5,
                             dropoutDesc::cudnnDropoutDescriptor = cudnnDropoutDescriptor(dropout),
                             xDesc::cudnnTensorDescriptor = TD(x),
                             yDesc::cudnnTensorDescriptor = xDesc,
                             y::R = similar(x),
                             reserveSpace::DevArray = cudnnDropoutReserveSpace(xDesc)
                             ) where {T,R<:DevArray{T}}
    if cudnnDropoutSeed[] >= 0
        # This is a very expensive call (40x dropout), so only use for debugging
        @warn "Knet.CUDNN.cudnnDropoutSeed[] >= 0: calling expensive cudnnSetDropoutDescriptor" maxlog=1
        @retry cudnnSetDropoutDescriptor(dropoutDesc, handle(), dropout, cudnnDropoutState[], sizeof(cudnnDropoutState[]), cudnnDropoutSeed[])
    end
    cudnnDropoutForward(handle(), dropoutDesc, xDesc, x, yDesc, y, reserveSpace, sizeof(reserveSpace))
    return y
end


function cudnnDropoutBackward(dy::R;
                              dropoutDesc::cudnnDropoutDescriptor,
                              dyDesc::cudnnTensorDescriptor,
                              dxDesc::cudnnTensorDescriptor,
                              reserveSpace::DevArray,
                              dx::R = similar(dy)
                              ) where {T,R<:DevArray{T}}
    cudnnDropoutBackward(handle(), dropoutDesc, dyDesc, dy, dxDesc, dx, reserveSpace, sizeof(reserveSpace))
    return dx
end


function cudnnDropoutReserveSpace(td::cudnnTensorDescriptor)
    # reserveSpace is ~1/8 of tensor size and passes info between forw and back
    rss = Csize_t[0]; cudnnDropoutGetReserveSpaceSize(td, rss)
    return CuArray{Int}(undef, (rss[1]-1)÷sizeof(Int)+1)
end


@primitive1((cudnnDropoutForward(x;
                                 dropout::Real = 0.5,
                                 dropoutDesc::cudnnDropoutDescriptor = cudnnDropoutDescriptor(dropout),
                                 xDesc::cudnnTensorDescriptor = TD(x),
                                 yDesc::cudnnTensorDescriptor = xDesc,
                                 y = similar(x),
                                 reserveSpace::DevArray = cudnnDropoutReserveSpace(xDesc)
                                 ),_dy,_y),
            cudnnDropoutBackward(_dy;
                                 dropoutDesc = dropoutDesc,
                                 dyDesc = xDesc,
                                 dxDesc = xDesc,
                                 reserveSpace = reserveSpace))

@primitive1 dropoutBackward(dy;o...)  throw(MethodError(back,dropoutBackward))

