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


cudnnOpTensor(x1,x2; o...)                 = cudnnOpTensorWithDefaults(x1,x2; o...)
cudnnOpTensor(x1,x2,opTensorDesc; o...)    = cudnnOpTensorWithDefaults(x1,x2; opTensorDesc, o...)
cudnnOpTensor!(y,x1,x2; o...)              = cudnnOpTensorWithDefaults(x1,x2; y, o...)
cudnnOpTensor!(y,x1,x2,opTensorDesc; o...) = cudnnOpTensorWithDefaults(x1,x2; y, opTensorDesc, o...)


# Compared to cudnnAddTensor!(copy(a),b), ~50% faster on (14,14,256,32)+(1,1,256,1), ~50% slower on (1,1,100,100)+(1,1,100,1)
# Unlike cudnnAddTensor it supports all broadcasting shapes up to ndims=5 as described in the documentation
function cudnnOpTensorWithDefaults(
    x1, x2;
    y = similar(x1),
    opTensorOp::cudnnOpTensorOp_t = CUDNN_OP_TENSOR_ADD,
    opTensorCompType::DataType = (eltype(x1) <: Float64 ? Float64 : Float32),
    opTensorNanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    opTensorDesc::cudnnOpTensorDescriptor = cudnnOpTensorDescriptor(opTensorOp, DT(opTensorCompType), opTensorNanOpt),
    alpha1::Real = 1,
    x1Desc::cudnnTensorDescriptor = TD(x1),
    alpha2::Real = 1,
    x2Desc::cudnnTensorDescriptor = TD(x2),
    beta::Real = 0,
    yDesc::cudnnTensorDescriptor = x1Desc
)
    @assert size(x1) == size(x2) == size(y)
    @assert eltype(x1) == eltype(x2) == eltype(y)
    @assert ndims(x1) <= 5
    alpha1, alpha2, beta = scalr(alpha1,x1), scalr(alpha2,x2), scalr(beta,y)
    cudnnOpTensorAutoGrad(x1, x2; opTensorDesc, alpha1, x1Desc, alpha2, x2Desc, beta, yDesc, y)
end


function cudnnOpTensorAutoGrad(x1, x2; opTensorDesc, alpha1, x1Desc, alpha2, x2Desc, beta, yDesc, y)
    CUDA.CUDNN.cudnnOpTensor(handle(), opTensorDesc, alpha1, x1Desc, x1, alpha2, x2Desc, x2, beta, yDesc, y)
    return y
end


# TODO: define backward function
