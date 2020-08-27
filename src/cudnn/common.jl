export TD, FD
import Base: unsafe_convert
using Base: size_to_strides
using Knet.KnetArrays: DevArray
using AutoGrad: Value

using CUDA.CUDNN: handle,
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

    cudnnFilterDescriptor_t,
    cudnnCreateFilterDescriptor,
    cudnnSetFilter4dDescriptor,
    cudnnGetFilter4dDescriptor,
    cudnnSetFilterNdDescriptor,
    cudnnGetFilterNdDescriptor,
    cudnnGetFilterSizeInBytes,
    cudnnDestroyFilterDescriptor,

    cudnnDataType_t,
        CUDNN_DATA_FLOAT,
        CUDNN_DATA_DOUBLE,
        CUDNN_DATA_HALF,
        CUDNN_DATA_INT8,
        CUDNN_DATA_INT32,
        CUDNN_DATA_INT8x4,
        CUDNN_DATA_UINT8,
        CUDNN_DATA_UINT8x4,
        CUDNN_DATA_INT8x32,

    cudnnTensorFormat_t,
        CUDNN_TENSOR_NCHW,
        CUDNN_TENSOR_NHWC,
        CUDNN_TENSOR_NCHW_VECT_C


# cudnn tensor/filter descriptors: need to use mutable structs in order to have finalizers.

mutable struct TensorDescriptor; ptr::cudnnTensorDescriptor_t; end
unsafe_convert(::Type{<:Ptr}, td::TensorDescriptor)=td.ptr
TD(a) = TD(eltype(a),size(a))
function TD(T::Type, dims::Dims{N}) where {N}
    ptr = cudnnTensorDescriptor_t[C_NULL]
    cudnnCreateTensorDescriptor(ptr)
    sz = Cint[reverse(dims)...]
    st = Cint[reverse(size_to_strides(1,dims...))...]
    cudnnSetTensorNdDescriptor(ptr[1], DT(T), N, sz, st)
    td = TensorDescriptor(ptr[1])
    finalizer(x->cudnnDestroyTensorDescriptor(x.ptr), td)
    return td
end


mutable struct FilterDescriptor; ptr::cudnnFilterDescriptor_t; end
unsafe_convert(::Type{<:Ptr}, fd::FilterDescriptor)=fd.ptr
FD(a) = FD(eltype(a),size(a),CUDNN_TENSOR_NCHW)
function FD(T::Type, dims::Dims{N},format::cudnnTensorFormat_t) where N
    ptr = cudnnFilterDescriptor_t[C_NULL]
    cudnnCreateFilterDescriptor(ptr)
    sz = Cint[reverse(dims)...]
    cudnnSetFilterNdDescriptor(ptr[1], DT(T), format, N, sz)
    fd = FilterDescriptor(ptr[1])
    finalizer(x->cudnnDestroyFilterDescriptor(x.ptr), fd)
    return fd
end


DT(::Type{T}) where T = get(DataTypes, T) do; error("CUDNN does not support $T"); end
DataTypes = Dict{Type,cudnnDataType_t}(
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

nothing
