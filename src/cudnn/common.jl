import CUDA
import Base: unsafe_convert
using CUDA: CU_NULL

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


macro cudnnDescriptor(x, set = Symbol("cudnnSet$(x)Descriptor"))
    sname = Symbol("cudnn$(x)Descriptor")
    tname = Symbol("cudnn$(x)Descriptor_t")
    cache = Symbol("cudnn$(x)DescriptorCache")
    create = Symbol("cudnnCreate$(x)Descriptor")
    destroy = Symbol("cudnnDestroy$(x)Descriptor")
    return quote
        mutable struct $sname; ptr::$tname; end
        Base.unsafe_convert(::Type{<:Ptr}, d::$sname)=d.ptr # needed for ccalls
        const $cache = Dict{Tuple,$sname}() # Dict is 3x faster than IdDict!
        function $sname(args...)
            get!($cache, args) do
                ptr = $tname[C_NULL]
                $create(ptr)
                $set(ptr[1], args...)
                d = $sname(ptr[1])
                finalizer(x->$destroy(x.ptr), d)
                return d
            end
        end
    end |> esc
end


@cudnnDescriptor(Tensor, cudnnSetTensorNdDescriptorEx)

const TD = cudnnTensorDescriptor  # short alias

function cudnnTensorDescriptor(
    a;
    format::cudnnTensorFormat_t=CUDNN_TENSOR_NCHW,
    dims::Vector{Cint}=dim4(size(a))
)
    @assert length(dims) <= CUDNN_DIM_MAX
    cudnnTensorDescriptor(format, DT(eltype(a)), Cint(length(dims)), dims)
end



@cudnnDescriptor(Filter, cudnnSetFilterNdDescriptor)

const FD = cudnnFilterDescriptor

function cudnnFilterDescriptor(
    a;
    format::cudnnTensorFormat_t=CUDNN_TENSOR_NCHW,
    dims::Vector{Cint}=dim4(size(a))
)
    @assert length(dims) <= CUDNN_DIM_MAX
    cudnnFilterDescriptor(DT(eltype(a)), format, Cint(length(dims)), dims)
end


# From cuDNN docs: Due to historical reasons, the minimum number of dimensions in the filter
# descriptor is three, and at most CUDNN_DIM_MAX dimensions (defined in cudnn.h =
# 8). However many operations only support 4 and 5. So we will pad dims to 4. TODO: check if this is ok with rnn and attn
dim4(s::Dims{0})=Cint[1,1,1,1]
dim4(s::Dims{1})=Cint[s[1],1,1,1]
dim4(s::Dims{2})=Cint[s[2],s[1],1,1]
dim4(s::Dims{3})=Cint[s[3],s[2],s[1],1]
dim4(s::Dims)   =Cint[reverse(s)...]


cudnnDataType(T::DataType) = get(cudnnDataTypeCache, T) do; error("CUDNN does not support $T"); end

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


# The scaling parameters are passed using a host memory pointer.
# The storage data types for alpha and beta are:
#    float for HALF and FLOAT tensors, and
#    double for DOUBLE tensors.
scalr(a,x)=Ref(convert(eltype(x)===Float64 ? Float64 : Float32, a))


cu_null(x) = (x === nothing ? CU_NULL : x)
c_null(x) = (x === nothing ? C_NULL : x)

