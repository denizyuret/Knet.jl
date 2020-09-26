using AutoGrad: AutoGrad, @primitive1, value

using CUDA.CUDNN:
    #cudnnConvolutionForward,
    cudnnConvolutionBackward,
    cudnnGetConvolutionNdForwardOutputDim,
    cudnnConvolutionDescriptor_t,
        cudnnCreateConvolutionDescriptor,
        cudnnSetConvolutionNdDescriptor,
        cudnnDestroyConvolutionDescriptor,
        cudnnConvolutionMode_t,
            CUDNN_CONVOLUTION,       # 0
            CUDNN_CROSS_CORRELATION, # 1
        cudnnNanPropagation_t,
            CUDNN_NOT_PROPAGATE_NAN, # 0
            CUDNN_PROPAGATE_NAN      # 1


"""
TODO

    cudnnConvolutionForward(x; mode, maxconvolutionNanOpt, nbDims, window, padding, stride, alpha, xDesc)
    cudnnConvolutionForward(x, d::cudnnConvolutionDescriptor; alpha, xDesc)
    cudnnConvolutionForward!(y, x; mode, maxconvolutionNanOpt, nbDims, window, padding, stride, alpha, beta, xDesc, yDesc)
    cudnnConvolutionForward!(y, x, d::cudnnConvolutionDescriptor; alpha, beta, xDesc, yDesc)

Return pooled `x`, overwriting `y` if provided, according to keyword arguments or the
convolution descriptor. Please see the [cuDNN
docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionForward) for
details.

The dimensions of `x,y` tensors that are less than 4-D are assumed to be padded on the left
with 1's. The first `n-2` are spatial dimensions, the last two are always assumed to be
channel and batch.

The arguments `window`, `padding`, and `stride` can be specified as `n-2` dimensional
vectors, tuples or a single integer which is assumed to be repeated `n-2` times. If any of
the entries is larger than the corresponding `x` dimension, the `x` dimension is used
instead.

Arguments:
* `mode = CUDNN_CONVOLUTION_MAX`
* `maxconvolutionNanOpt = CUDNN_NOT_PROPAGATE_NAN`
* `window = 2`
* `padding = 0`
* `stride = window`
* `alpha = 1`
* `beta = 0`
* `xDesc = cudnnTensorDescriptor(x)`
* `yDesc = cudnnTensorDescriptor(y)`

"""
cudnnConvolutionForward, cudnnConvolutionForward!


function cudnnConvolutionForward(
    w::DevArray{T,N},           # TODO: w,x or x,w? libcudnn is x,w,y
    x::DevArray{T,N};
    padding::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 0,
    stride::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 1,
    dilation::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 1,
    group::Integer = 1,         # TODO: special constructor for descriptor
    mode::cudnnConvolutionMode_t = CUDNN_CONVOLUTION,
    dataType::cudnnDataType_t = CUDNN_DATA_FLOAT, # see cudnn docs, TODO: test with 16,32,64
    alpha::Real = 1,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x),
    wDesc::cudnnFilterDescriptor = cudnnFilterDescriptor(w),
    algo, workSpace # TODO: which method sets these?
) where {T,N}
    arrayLength = max(2, ndims(x)-2)
    convolutionDesc = cudnnConvolutionDescriptor(mode, maxconvolutionNanOpt, Cint(nbDims), pooldims(window,size(x)), pooldims(padding,size(x)), pooldims(stride,size(x)))
    cudnnConvolutionForward(x, convolutionDesc; alpha, xDesc)
end


function cudnnConvolutionForward!(
    y::DevArray{T,N},
    x::DevArray{T,N};
    mode::cudnnConvolutionMode_t = CUDNN_CONVOLUTION_MAX,
    maxconvolutionNanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    window::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 2,
    padding::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 0,
    stride::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = window,
    alpha::Real = 1,
    beta::Real = 0,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x),
    yDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(y)
) where {T,N}
    nbDims = max(2, ndims(x)-2)
    convolutionDesc = cudnnConvolutionDescriptor(mode, maxconvolutionNanOpt, Cint(nbDims), pooldims(window,size(x)), pooldims(padding,size(x)), pooldims(stride,size(x)))
    cudnnConvolutionForward!(y, x, convolutionDesc; alpha, beta, xDesc, yDesc)
end


function cudnnConvolutionForward(
    x::DevArray{T,N},
    convolutionDesc::cudnnConvolutionDescriptor;
    alpha::Real = 1,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x)
) where {T,N}
    d = Array{Cint}(undef, max(4, ndims(x)))
    cudnnGetConvolutionNdForwardOutputDim(convolutionDesc, xDesc, length(d), d)
    if length(d) > ndims(x) # This happens when x is (X,C,N), its TD is [N,C,X,1]
        @assert all(d[ndims(x)+1:end] .== 1)
        d = d[1:ndims(x)]
    end
    y = similar(x, reverse(d)...)
    yDesc = cudnnTensorDescriptor(y)
    beta = 0
    cudnnConvolutionForward!(y, x, convolutionDesc; alpha, beta, xDesc, yDesc)
end


function cudnnConvolutionForward!(
    y::DevArray{T,N},
    x::DevArray{T,N},
    convolutionDesc::cudnnConvolutionDescriptor;
    alpha::Real = 1,
    beta::Real = 0,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x),
    yDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(y)
) where {T,N}
    alpha, beta = scalr(alpha,x), scalr(beta,x)
    _cudnnConvolutionForward(x; convolutionDesc, alpha, xDesc, beta, yDesc, y)
end


# This intermediate function is designed to make gradient definition easier.
# The only main args shoudl be the gradient targets. All kwargs are mandatory.
# Args not used by cudnnConvolutionBackward can be dropped.
function _cudnnConvolutionForward(x; convolutionDesc, alpha, xDesc, beta, yDesc, y)
    CUDA.CUDNN.cudnnConvolutionForward(handle(), convolutionDesc, alpha, xDesc, x, beta, yDesc, y)
    return y
end


# Define gradients
@primitive1((_cudnnConvolutionForward(x; convolutionDesc, alpha, xDesc, beta, yDesc, y),
             _dy,_y),
            ((x,y,dy,dx) = (value(x),value(_y),value(_dy),similar(x));
             cudnnConvolutionBackward(handle(), convolutionDesc, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx);
             dx))


# Convert the integer, tuple or array to convolution dims compatible with array size
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
