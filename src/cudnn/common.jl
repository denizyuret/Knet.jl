import CUDA
using CUDA: CU_NULL
using CUDA.CUDNN:
    cudnnDataType_t,
        CUDNN_DATA_FLOAT,   # 0,
        CUDNN_DATA_DOUBLE,  # 1,
        CUDNN_DATA_HALF,    # 2,
        CUDNN_DATA_INT8,    # 3,
        CUDNN_DATA_INT32,   # 4,
        CUDNN_DATA_INT8x4,  # 5,
        CUDNN_DATA_UINT8,   # 6,
        CUDNN_DATA_UINT8x4, # 7,
        CUDNN_DATA_INT8x32  # 8,


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


cudnnDataType(::Type{Float32}) = CUDNN_DATA_FLOAT
cudnnDataType(::Type{Float64}) = CUDNN_DATA_DOUBLE
cudnnDataType(::Type{Float16}) = CUDNN_DATA_HALF
cudnnDataType(::Type{Int8}) = CUDNN_DATA_INT8
cudnnDataType(::Type{UInt8}) = CUDNN_DATA_UINT8
cudnnDataType(::Type{Int32}) = CUDNN_DATA_INT32
# The following are 32-bit elements each composed of 4 8-bit integers, only supported with CUDNN_TENSOR_NCHW_VECT_C (TODO)
# CUDNN_DATA_INT8x4,
# CUDNN_DATA_UINT8x4,
# CUDNN_DATA_INT8x32,

const DT = cudnnDataType


# The scaling parameters are passed using a host memory pointer.
# The storage data types for alpha and beta are:
#    float for HALF and FLOAT tensors, and
#    double for DOUBLE tensors.
scalr(a,x)=Ref(convert(eltype(x)===Float64 ? Float64 : Float32, a))


# Convert nothing to CU_NULL or C_NULL for ccall:
cu_null(x) = (x === nothing ? CU_NULL : x)
c_null(x) = (x === nothing ? C_NULL : x)


# Create temporary workspace. Use 128 to avoid alignment issues.
function cudnnWorkspace(nbytes)
    nbytes == 0 ? nothing : CuArray{Int128}(undef, (nbytes-1)÷sizeof(Int128)+1)
end
