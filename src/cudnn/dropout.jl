using Knet.KnetArrays: DevArray
using AutoGrad: AutoGrad, @primitive1, recording
using CUDA: CuArray

using CUDA.CUDNN: 
   #cudnnDropoutForward,
    cudnnDropoutBackward,
    cudnnDropoutDescriptor_t,
        cudnnCreateDropoutDescriptor,
        cudnnSetDropoutDescriptor,
        cudnnGetDropoutDescriptor,
        cudnnRestoreDropoutDescriptor,
        cudnnDestroyDropoutDescriptor,
    cudnnDropoutGetStatesSize,
    cudnnDropoutGetReserveSpaceSize,
    handle


cudnnDropoutForward(x; o...)                  = cudnnDropoutForwardWithDefaults(x; o...)
cudnnDropoutForward(x, dropoutDesc; o...)     = cudnnDropoutForwardWithDefaults(x; dropoutDesc, o...)
cudnnDropoutForward!(y, x; o...)              = cudnnDropoutForwardWithDefaults(x; y, o...)
cudnnDropoutForward!(y, x, dropoutDesc; o...) = cudnnDropoutForwardWithDefaults(x; y, dropoutDesc, o...)


function cudnnDropoutForwardWithDefaults(
    x;
    y = similar(x),
    dropout::Real = 0.5,
    dropoutDesc::cudnnDropoutDescriptor = cudnnDropoutDescriptor(dropout),
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x),
    yDesc::cudnnTensorDescriptor = xDesc,
    reserveSpace::DevArray = cudnnDropoutReserveSpace(xDesc)
)
    cudnnDropoutForwardAutoGrad(x; xDesc, y, yDesc, dropout, dropoutDesc, reserveSpace)
end


function cudnnDropoutForwardAutoGrad(x; xDesc, y, yDesc, dropout, dropoutDesc, reserveSpace)
    if !recording()
        @warn "cudnnDropoutForward called outside of training." maxlog=1
    end
    if cudnnDropoutSeed[] >= 0
        # This is a very expensive call (40x dropout), so only use for debugging
        @warn "Knet.CUDNN.cudnnDropoutSeed[] >= 0: calling expensive cudnnSetDropoutDescriptor" maxlog=1
        @retry cudnnSetDropoutDescriptor(dropoutDesc, handle(), dropout, cudnnDropoutState[], sizeof(cudnnDropoutState[]), cudnnDropoutSeed[])
    end
    CUDA.CUDNN.cudnnDropoutForward(handle(), dropoutDesc, xDesc, x, yDesc, y, reserveSpace, sizeof(reserveSpace))
    return y
end


@primitive1((cudnnDropoutForwardAutoGrad(x; xDesc, y, yDesc, dropout, dropoutDesc, reserveSpace),
             _dy, _y),
            ((x,y,dy,dx) = (value(x),value(_y),value(_dy),similar(x));
             cudnnDropoutBackward(handle(), dropoutDesc, yDesc, dy, xDesc, dx, reserveSpace, sizeof(reserveSpace));
             dx))


function cudnnDropoutReserveSpace(td::cudnnTensorDescriptor)
    # reserveSpace is ~1/8 of tensor size and passes info between forw and back
    rss = Csize_t[0]; cudnnDropoutGetReserveSpaceSize(td, rss)
    return CuArray{Int128}(undef, (rss[1]-1)÷sizeof(Int128)+1)
end


# cudnnDropoutForward() doc says:
# "This function should not be running concurrently with another cudnnDropoutForward() function using the same states."
# So I am going to assume using a single buffer is fine until we have to deal with concurrency.
# TODO: fix this based on default_rng in Random
cudnnDropoutState = Ref{CuArray{Int128,1}}()

# To debug gradients set cudnnDropoutSeed[] >= 0 which makes all dropout operations deterministic
cudnnDropoutSeed = Ref{Int}(-1)


const DD = cudnnDropoutDescriptor

cudnnDropoutDescriptor(x) = cudnnDropoutDescriptor(convert(Cfloat,x), nothing) # fixes ambiguity issue

function cudnnSetDropoutDescriptorFromFloat(ptr::cudnnDropoutDescriptor_t, dropout::Cfloat, ignore::Nothing)
    if !isassigned(cudnnDropoutState)
        ssize = Csize_t[0]; cudnnDropoutGetStatesSize(handle(), ssize)
        cudnnDropoutState[] = CuArray{Int128}(undef, (ssize[1]-1)÷sizeof(Int128)+1)
    end
    seed = floor(Culonglong,time())
    @retry cudnnSetDropoutDescriptor(ptr, handle(), dropout, cudnnDropoutState[], sizeof(cudnnDropoutState[]), seed)
end
