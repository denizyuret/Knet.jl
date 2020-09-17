using AutoGrad: AutoGrad, @primitive1, value

using CUDA.CUDNN:
    #cudnnPoolingForward,
    cudnnPoolingBackward,
    cudnnGetPoolingNdForwardOutputDim,
    cudnnPoolingDescriptor_t,
        cudnnCreatePoolingDescriptor,
        cudnnSetPoolingNdDescriptor,
        cudnnDestroyPoolingDescriptor,
        cudnnPoolingMode_t,
            CUDNN_POOLING_MAX,                           # 0,
            CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, # 1, /* count for average includes padded values */
            CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, # 2, /* count for average does not include padded values */
            CUDNN_POOLING_MAX_DETERMINISTIC,             # 3
        cudnnNanPropagation_t,
            CUDNN_NOT_PROPAGATE_NAN, # 0
            CUDNN_PROPAGATE_NAN      # 1


function cudnnPoolingForward(
    x, y = nothing;
    mode::cudnnPoolingMode_t = CUDNN_POOLING_MAX,
    maxpoolingNanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    nbDims::Integer = max(2, ndims(x)-2),
    window::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 2,
    padding::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 0,
    stride::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = window,
    poolingDesc::cudnnPoolingDescriptor = cudnnPoolingDescriptor(mode, maxpoolingNanOpt, Cint(nbDims), pooldims(window,size(x)), pooldims(padding,size(x)), pooldims(stride,size(x))),
    alpha::Real = 1,
    beta::Real = 0,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x),
    yDesc::Union{Nothing,cudnnTensorDescriptor} = nothing
)
    alpha, beta = scalr(alpha,x), scalr(beta,x)
    if y === nothing
        d = Array{Cint}(undef, nbDims+2)
        cudnnGetPoolingNdForwardOutputDim(poolingDesc, xDesc, nbDims+2, d)
        if length(d) > ndims(x) # This happens when x is (X,C,N), its TD is [N,C,X,1]
            @assert all(d[ndims(x)+1:end] .== 1)
            d = d[1:ndims(x)]
        end
        y = similar(x, reverse(d)...)
        yDesc = cudnnTensorDescriptor(y)
    end
    _cudnnPoolingForward(x; poolingDesc, alpha, xDesc, beta, yDesc, y)
end
    

# Convert the integer, tuple or array to pooling dims compatible with array size
function pooldims(d, s::Dims{N}) where N
    if d isa Integer || length(d) == N-2
        Cint[reverse(min.(d,s[1:N-2]))...]
    else
        throw(DimensionMismatch("Cannot pool $(Base.dims2string(s)) array with $d pooldims."))
    end
end

pooldims(d, s::Dims{3}) = pooldims(d, (1,s...))
pooldims(d, s::Dims{2}) = pooldims(d, (1,1,s...))
pooldims(d, s::Dims{1}) = pooldims(d, (1,1,1,s...))
pooldims(d, s::Dims{0}) = pooldims(d, (1,1,1,1))


# This intermediate function is designed to make gradient definition easier:
# The only main args are the gradient targets.
# All kwargs are mandatory. Ones not used by cudnnPoolingBackward can be dropped.
function _cudnnPoolingForward(x; poolingDesc, alpha, xDesc, beta, yDesc, y)
    CUDA.CUDNN.cudnnPoolingForward(handle(), poolingDesc, alpha, xDesc, x, beta, yDesc, y)
    return y
end


# Define gradients
@primitive1((_cudnnPoolingForward(x; poolingDesc, alpha, xDesc, beta, yDesc, y),
             _dy,_y),
            ((x,y,dy,dx) = (value(x),value(_y),value(_dy),similar(x));
             cudnnPoolingBackward(handle(), poolingDesc, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx);
             dx))
