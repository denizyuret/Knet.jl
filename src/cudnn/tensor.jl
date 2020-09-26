using CUDA.CUDNN: 
    cudnnTensorDescriptor_t,
        cudnnCreateTensorDescriptor,
        cudnnSetTensor4dDescriptor,
        cudnnSetTensor4dDescriptorEx,
        cudnnGetTensor4dDescriptor,
        cudnnSetTensorNdDescriptor,
        cudnnSetTensorNdDescriptorEx,
        cudnnGetTensorNdDescriptor,
        cudnnGetTensorSizeInBytes,
        cudnnDestroyTensorDescriptor,
        CUDNN_DIM_MAX,
    cudnnFilterDescriptor_t,
        cudnnCreateFilterDescriptor,
        cudnnSetFilter4dDescriptor,
        cudnnGetFilter4dDescriptor,
        cudnnSetFilterNdDescriptor,
        cudnnGetFilterNdDescriptor,
        cudnnGetFilterSizeInBytes,
        cudnnDestroyFilterDescriptor,
    cudnnTensorFormat_t,
        CUDNN_TENSOR_NCHW,        # 0, /* row major (wStride = 1, hStride = w) */
        CUDNN_TENSOR_NHWC,        # 1, /* feature maps interleaved ( cStride = 1 )*/
        CUDNN_TENSOR_NCHW_VECT_C  # 2, /* each image point is vector of element of C, vector length in data type */


const TD = cudnnTensorDescriptor  # short alias

function cudnnTensorDescriptor(   # constructor from array
    array;
    format::cudnnTensorFormat_t=CUDNN_TENSOR_NCHW,
    dims::Vector{Cint}=dim4(size(array))
)
    @assert length(dims) <= CUDNN_DIM_MAX
    cudnnTensorDescriptor(format, DT(eltype(array)), Cint(length(dims)), dims)
end


const FD = cudnnFilterDescriptor # short alias

function cudnnFilterDescriptor(  # constructor from array
    array;
    format::cudnnTensorFormat_t=CUDNN_TENSOR_NCHW,
    dims::Vector{Cint}=dim4(size(array))
)
    @assert length(dims) <= CUDNN_DIM_MAX
    cudnnFilterDescriptor(DT(eltype(array)), format, Cint(length(dims)), dims)
end


# From cuDNN docs: Due to historical reasons, the minimum number of dimensions in the filter
# descriptor is three, and at most CUDNN_DIM_MAX dimensions (defined in cudnn.h = 8). 
# However many operations only support 4 and 5. So we will pad dims to 4. 
# TODO: check if this is ok with rnn and attn
dim4(s::Dims{0})=Cint[1,1,1,1]
dim4(s::Dims{1})=Cint[s[1],1,1,1]
dim4(s::Dims{2})=Cint[s[2],s[1],1,1]
dim4(s::Dims{3})=Cint[s[3],s[2],s[1],1]
dim4(s::Dims)   =Cint[reverse(s)...]


# If array is nothing, return nothing for descriptor
cudnnTensorDescriptor(::Nothing; o...) = nothing
cudnnFilterDescriptor(::Nothing; o...) = nothing
