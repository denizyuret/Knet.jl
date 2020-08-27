export dropoutForward
import Base: unsafe_convert
using Knet.KnetArrays: DevArray
using AutoGrad: AutoGrad, @primitive1
using CUDA: CuArray, CuPtr, CU_NULL

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


@primitive1 dropoutForward(x; reserveSpace=Ref{Any}(), dropoutDesc=GLOBAL_DD, o...),dy,y  dropoutBackward(dy; reserveSpace=reserveSpace, dropoutDesc=dropoutDesc)
@primitive1 dropoutBackward(dy;o...)  throw(MethodError(back,dropoutBackward))

# cudnnDropoutForward() doc says:
# "This function should not be running concurrently with another cudnnDropoutForward() function using the same states."
# So I am going to assume using a single buffer/descriptor is fine until we have to deal with concurrency.
# TODO: model this based on default_rng in Random
mutable struct DropoutDescriptor; ptr; states; end
GLOBAL_DD = Ref{DropoutDescriptor}()


function dropoutForward(x::R; dropout::Real=0.5, seed::Integer=-1,
                        dropoutDesc=GLOBAL_DD, reserveSpace=Ref{Any}()
                        ) where {T,R<:DevArray{T}}
    y, td = similar(x), TD(T, (1,1,length(x),1))
    if !isassigned(dropoutDesc)
        dropoutDesc[] = DD(dropout=dropout)
    end
    (dd_dropout, dd_states, dd_seed) = getDropoutDescriptor(dropoutDesc[])
    if !(dropout ≈ dd_dropout) || seed >= 0
        #@show (dropout, dd_dropout, seed, dd_seed)
        seed = (seed >= 0 ? Unsigned(seed) : dd_seed)
        # This is a very expensive call (10x dropout), so rethink interface
        @cudnn_retry unsafe_cudnnSetDropoutDescriptor(dropoutDesc[], handle(), dropout, dd_states, sizeof(dropoutDesc[].states), seed)
    end
    if !isassigned(reserveSpace)
        rss = Csize_t[0]; cudnnDropoutGetReserveSpaceSize(td, rss)
        reserveSpace[] = CuArray{Int}(undef, (rss[1]-1)÷sizeof(Int)+1)
    end
    cudnnDropoutForward(handle(), dropoutDesc[], td, x, td, y, reserveSpace[], sizeof(reserveSpace[]))
    return y
end


function dropoutBackward(dy::R; dropoutDesc, reserveSpace) where {T,R<:DevArray{T}}
    #@show summary.((dy,dropoutDesc[],reserveSpace[]))
    dx, td = similar(dy), TD(T, (1,1,length(dy),1))
    cudnnDropoutBackward(handle(), dropoutDesc[], td, dy, td, dx, reserveSpace[], sizeof(reserveSpace[]))
    return dx
end


# TODO: rewrite when CUDA.jl fixed
using CUDA.CUDNN: @runtime_ccall, libcudnn, cudnnStatus_t, cudnnHandle_t
function getDropoutDescriptor(dropoutDesc::DropoutDescriptor)
    dropout = Cfloat[0]
    states = CuPtr{Cvoid}[CU_NULL]
    seed = Culonglong[0]
    # This is broken as of CUDA.jl 1.3.3
    # cudnnGetDropoutDescriptor(dd, handle(), dropout, states, seed)
    @runtime_ccall((:cudnnGetDropoutDescriptor, libcudnn()), cudnnStatus_t,
                   (cudnnDropoutDescriptor_t, cudnnHandle_t, Ptr{Cfloat},
                    Ptr{CuPtr{Cvoid}}, Ptr{Culonglong}),
                   dropoutDesc, handle(), dropout, states, seed)
    (dropout[1], states[1], seed[1])
end


unsafe_convert(::Type{<:Ptr}, dd::DropoutDescriptor)=dd.ptr
function DD(; dropout::Real=0.5, seed::Unsigned=floor(Culonglong,time()))
    sptr = Csize_t[0]
    cudnnDropoutGetStatesSize(handle(), sptr)
    states = CuArray{Int}(undef, (sptr[1]-1)÷sizeof(Int)+1)
    ptr = cudnnDropoutDescriptor_t[C_NULL]
    cudnnCreateDropoutDescriptor(ptr)
    @cudnn_retry unsafe_cudnnSetDropoutDescriptor(ptr[1], handle(), dropout, states, sizeof(states), seed)
    dd = DropoutDescriptor(ptr[1], states)
    finalizer(x->cudnnDestroyDropoutDescriptor(x.ptr), dd)
    return dd
end
    

