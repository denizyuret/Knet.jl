import Base: unsafe_convert
using Knet.KnetArrays: DevArray

using CUDA.CUDNN:
    #cudnnOpTensor
    cudnnOpTensorDescriptor_t,
        cudnnCreateOpTensorDescriptor,
        cudnnSetOpTensorDescriptor,
        cudnnGetOpTensorDescriptor,
        cudnnDestroyOpTensorDescriptor,
    cudnnOpTensorOp_t,
        CUDNN_OP_TENSOR_ADD,  # 0,
        CUDNN_OP_TENSOR_MUL,  # 1,
        CUDNN_OP_TENSOR_MIN,  # 2,
        CUDNN_OP_TENSOR_MAX,  # 3,
        CUDNN_OP_TENSOR_SQRT, # 4, performed only on first arg
        CUDNN_OP_TENSOR_NOT,  # 5, performed only on first arg
    handle


# Compared to cudnnAddTensor!(copy(a),b), ~50% faster on (14,14,256,32)+(1,1,256,1), ~50% slower on (1,1,100,100)+(1,1,100,1)
# Unlike cudnnAddTensor it supports all broadcasting shapes up to ndims=5 as described in the documentation
function cudnnOpTensor(
    x1::R, x2::R, y::R = similar(x1);
    opTensorOp::cudnnOpTensorOp_t = CUDNN_OP_TENSOR_ADD,
    opTensorCompType::DataType = (T <: Float64 ? Float64 : Float32),
    opTensorNanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    opTensorDesc::cudnnOpTensorDescriptor = cudnnOpTensorDescriptor(opTensorOp, DT(opTensorCompType), opTensorNanOpt),
    alpha1::Real = 1,
    x1Desc::cudnnTensorDescriptor = TD(x1),
    alpha2::Real = 1,
    x2Desc::cudnnTensorDescriptor = TD(x2),
    beta::Real = 0,
    yDesc::cudnnTensorDescriptor = x1Desc
) where {T,N,R<:DevArray{T,N}}
    @assert N <= 5
    cudnnOpTensor(handle(), opTensorDesc, Ref(T(alpha1)), x1Desc, x1, Ref(T(alpha2)), x2Desc, x2, Ref(T(beta)), yDesc, y)
    return y
end
