import CUDA
import Base: unsafe_convert

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
    cudnnDataType_t,
        CUDNN_DATA_FLOAT,   # 0,
        CUDNN_DATA_DOUBLE,  # 1,
        CUDNN_DATA_HALF,    # 2,
        CUDNN_DATA_INT8,    # 3,
        CUDNN_DATA_INT32,   # 4,
        CUDNN_DATA_INT8x4,  # 5,
        CUDNN_DATA_UINT8,   # 6,
        CUDNN_DATA_UINT8x4, # 7,
        CUDNN_DATA_INT8x32, # 8,
    cudnnTensorFormat_t,
        CUDNN_TENSOR_NCHW,        # 0, /* row major (wStride = 1, hStride = w) */
        CUDNN_TENSOR_NHWC,        # 1, /* feature maps interleaved ( cStride = 1 )*/
        CUDNN_TENSOR_NCHW_VECT_C, # 2, /* each image point is vector of element of C, vector length in data type */
    cudnnNanPropagation_t,
        CUDNN_NOT_PROPAGATE_NAN, # 0,
        CUDNN_PROPAGATE_NAN,     # 1,
    handle


mutable struct cudnnTensorDescriptor; ptr::cudnnTensorDescriptor_t; end # Has to be mutable to have a finalizer

const cudnnTensorDescriptorCache = Dict{Tuple,cudnnTensorDescriptor}() # Dict is 3x faster than IdDict!

unsafe_convert(::Type{<:Ptr}, td::cudnnTensorDescriptor)=td.ptr # needed for ccalls

cudnnTensorDescriptor(a) = cudnnTensorDescriptor(eltype(a),size(a),CUDNN_TENSOR_NCHW)

const TD = cudnnTensorDescriptor  # short alias

function cudnnTensorDescriptor(T::DataType, size::Dims{N}, format::cudnnTensorFormat_t) where N
    get!(cudnnTensorDescriptorCache,(T,size,format)) do
        @assert N <= CUDNN_DIM_MAX
        if N < 4; size = pad4(size); end
        ptr = cudnnTensorDescriptor_t[C_NULL]
        cudnnCreateTensorDescriptor(ptr)
        sz = Cint[reverse(size)...]
        cudnnSetTensorNdDescriptorEx(ptr[1], format, DT(T), length(sz), sz)
        td = cudnnTensorDescriptor(ptr[1])
        finalizer(x->cudnnDestroyTensorDescriptor(x.ptr), td)
        return td
    end
end


mutable struct cudnnFilterDescriptor; ptr::cudnnFilterDescriptor_t; end

const cudnnFilterDescriptorCache = Dict{Tuple,cudnnFilterDescriptor}()

unsafe_convert(::Type{<:Ptr}, fd::cudnnFilterDescriptor)=fd.ptr

cudnnFilterDescriptor(a) = cudnnFilterDescriptor(eltype(a),size(a),CUDNN_TENSOR_NCHW)

const FD = cudnnFilterDescriptor

function cudnnFilterDescriptor(T::DataType, size::Dims{N}, format::cudnnTensorFormat_t) where N
    get!(cudnnFilterDescriptorCache, (T, size, format)) do
        @assert N <= CUDNN_DIM_MAX
        if N < 4; size = pad4(size); end
        ptr = cudnnFilterDescriptor_t[C_NULL]
        cudnnCreateFilterDescriptor(ptr)
        sz = Cint[reverse(size)...]
        cudnnSetFilterNdDescriptor(ptr[1], DT(T), format, length(sz), sz)
        fd = cudnnFilterDescriptor(ptr[1])
        finalizer(x->cudnnDestroyFilterDescriptor(x.ptr), fd)
        return fd
    end
end


# From cuDNN docs: Due to historical reasons, the minimum number of dimensions in the filter
# descriptor is three, and at most CUDNN_DIM_MAX dimensions (defined in cudnn.h =
# 8). However many operations only support 4 and 5. So we will pad dims to 4. TODO: check if this is ok with rnn and attn
pad4(s::Dims{0})=(1,1,1,1)
pad4(s::Dims{1})=(1,1,1,s...)
pad4(s::Dims{2})=(1,1,s...)
pad4(s::Dims{3})=(1,s...)


cudnnDataType(::Type{T}) where T = get(cudnnDataTypeCache, T) do; error("CUDNN does not support $T"); end

const DT = cudnnDataType

const cudnnDataTypeCache = Dict{DataType,cudnnDataType_t}(
    Float32 => CUDNN_DATA_FLOAT,
    Float64 => CUDNN_DATA_DOUBLE,
    Float16 => CUDNN_DATA_HALF,
    Int8 => CUDNN_DATA_INT8,
    UInt8 => CUDNN_DATA_UINT8,
    Int32 => CUDNN_DATA_INT32,
    # The following are 32-bit elements each composed of 4 8-bit integers, only supported with CUDNN_TENSOR_NCHW_VECT_C (TODO)
    # CUDNN_DATA_INT8x4,
    # CUDNN_DATA_UINT8x4,
    # CUDNN_DATA_INT8x32,
)


"Repeat low level cudnn calls that fail due to memory issues"
macro retry(ex)
    @assert Meta.isexpr(ex, :call)
    @assert ex.args[1] isa Symbol
    unsafe_ex = Expr(:call, Meta.parse("CUDA.CUDNN.unsafe_$(ex.args[1])"), ex.args[2:end]...)
    quote
        res = CUDA.CUDNN.@retry_reclaim x->(x ∈ (CUDA.CUDNN.CUDNN_STATUS_ALLOC_FAILED, CUDA.CUDNN.CUDNN_STATUS_EXECUTION_FAILED)) begin
            $unsafe_ex
        end 
        if res != CUDA.CUDNN.CUDNN_STATUS_SUCCESS
            CUDA.CUDNN.throw_api_error(res)
        end
    end |> esc
end

