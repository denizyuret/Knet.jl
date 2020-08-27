export multiHeadAttnForward
import Base: unsafe_convert
using Knet.KnetArrays: DevArray
using AutoGrad: AutoGrad, @primitive1

using CUDA.CUDNN: handle,
    cudnnMultiHeadAttnForward,
    cudnnMultiHeadAttnBackwardData,
    cudnnMultiHeadAttnBackwardWeights,
    cudnnAttnDescriptor_t,
        cudnnCreateAttnDescriptor,
        cudnnDestroyAttnDescriptor,
        cudnnSetAttnDescriptor,
        cudnnGetAttnDescriptor,
        CUDNN_ATTN_QUERYMAP_ALL_TO_ONE,
        CUDNN_ATTN_QUERYMAP_ONE_TO_ONE,
        CUDNN_ATTN_DISABLE_PROJ_BIASES,
        CUDNN_ATTN_ENABLE_PROJ_BIASES,
        cudnnDataType_t,
        cudnnMathType_t,
        cudnnDropoutDescriptor_t,
    cudnnMathType_t,
        CUDNN_DEFAULT_MATH,
        CUDNN_TENSOR_OP_MATH,
        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
    cudnnSeqDataDescriptor_t,
    cudnnMultiHeadAttnWeightKind_t,
    cudnnGetMultiHeadAttnBuffers,
    cudnnGetMultiHeadAttnWeights,
    cudnnAttnQueryMap_t
    

@primitive1 multiHeadAttnForward(x; o...),dy,y  multiHeadAttnBackwardWeights(x,y,dy; o...)  multiHeadAttnBackwardData(x,y,dy; o...)
@primitive1 multiHeadAttnBackwardData(x,y...;o...)  throw(MethodError(back,multiHeadAttnBackwardData))
@primitive1 multiHeadAttnBackwardWeights(x,y...;o...)  throw(MethodError(back,multiHeadAttnBackwardWeights))


function multiHeadAttnForward
end

function multiHeadAttnBackwardData
end

function multiHeadAttnBackwardWeights
end

mutable struct AttnDescriptor; ptr; end
unsafe_convert(::Type{<:Ptr}, mha::AttnDescriptor)=mha.ptr
function MHA(; # Defaults:
             attnMode::Unsigned = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | CUDNN_ATTN_DISABLE_PROJ_BIASES,
             nHeads::Integer,
             smScaler::Real = 1.0,
             dataType::DataType,
             computePrec::DataType = dataType,
             mathType::cudnnMathType_t = (dataType === Float16 ? CUDNN_TENSOR_OP_MATH :
                                          dataType === Float32 ? CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION :
                                          CUDNN_DEFAULT_MATH),
             
             )
    ptr = cudnnAttnDescriptor_t[C_NULL]
    cudnnCreataAttnDescriptor(ptr)
    cudnnSetAttnDescriptor(ptr[1], mode, nHeads, DT(dataType), DT(computePrec))
    mha = AttnDescriptor(ptr[1])
    finalizer(x->cudnnDestroyAttnDescriptor(x.ptr), mha)
    return mha
end

