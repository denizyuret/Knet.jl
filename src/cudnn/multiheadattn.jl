import Base: unsafe_convert
using Knet.KnetArrays: DevArray
using AutoGrad: AutoGrad, @primitive1

import CUDA.CUDNN: 
    cudnnMultiHeadAttnForward,
    cudnnMultiHeadAttnBackwardData,
    cudnnMultiHeadAttnBackwardWeights
using CUDA.CUDNN: 
    cudnnGetMultiHeadAttnBuffers,
    cudnnGetMultiHeadAttnWeights,
    cudnnAttnDescriptor_t,
        cudnnCreateAttnDescriptor,
        cudnnDestroyAttnDescriptor,
        cudnnSetAttnDescriptor,
        cudnnGetAttnDescriptor,
        cudnnDataType_t,
        cudnnDropoutDescriptor_t,
    cudnnSeqDataDescriptor_t,
        cudnnCreateSeqDataDescriptor,
        cudnnDestroySeqDataDescriptor,
        cudnnSetSeqDataDescriptor,
        cudnnGetSeqDataDescriptor,
    cudnnSeqDataAxis_t,
        CUDNN_SEQDATA_TIME_DIM,  # 0, /* index in time */
        CUDNN_SEQDATA_BATCH_DIM, # 1, /* index in batch */
        CUDNN_SEQDATA_BEAM_DIM,  # 2, /* index in beam */
        CUDNN_SEQDATA_VECT_DIM,  # 3  /* index in vector */
    cudnnAttnQueryMap_t,
        CUDNN_ATTN_QUERYMAP_ALL_TO_ONE, # 0         /* multiple Q-s map to a single (K,V) set when beam size > 1 */
        CUDNN_ATTN_QUERYMAP_ONE_TO_ONE, # (1U << 0) /* multiple Q-s map to multiple (K,V) sets when beam size > 1 */
        CUDNN_ATTN_DISABLE_PROJ_BIASES, # 0         /* no biases in attention input and output projections */
        CUDNN_ATTN_ENABLE_PROJ_BIASES,  # (1U << 1) /* use biases in attention input and output projections */
    cudnnMultiHeadAttnWeightKind_t,
        CUDNN_MH_ATTN_Q_WEIGHTS, # 0, /* input projection weights for 'queries' */
        CUDNN_MH_ATTN_K_WEIGHTS, # 1, /* input projection weights for 'keys' */
        CUDNN_MH_ATTN_V_WEIGHTS, # 2, /* input projection weights for 'values' */
        CUDNN_MH_ATTN_O_WEIGHTS, # 3, /* output projection weights */
        CUDNN_MH_ATTN_Q_BIASES,  # 4, /* input projection bias tensor for 'queries' */
        CUDNN_MH_ATTN_K_BIASES,  # 5, /* input projection bias for 'keys' */
        CUDNN_MH_ATTN_V_BIASES,  # 6, /* input projection bias for 'values' */
        CUDNN_MH_ATTN_O_BIASES,  # 7, /* output projection biases */
    cudnnMathType_t,
        CUDNN_DEFAULT_MATH,                    # 0,
        CUDNN_TENSOR_OP_MATH,                  # 1,
        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION, # 2,
        CUDNN_FMA_MATH,                        # 3,
    handle
    

mutable struct cudnnAttnDescriptor; ptr::cudnnAttnDescriptor_t; end

unsafe_convert(::Type{<:Ptr}, mha::cudnnAttnDescriptor)=mha.ptr

const cudnnAttnDescriptorCache = Dict{Tuple{},cudnnAttnDescriptor}()

function cudnnAttnDescriptor(args...)
    get!(cudnnAttnDescriptorCache, args) do
        ptr = cudnnAttnDescriptor_t[C_NULL]
        cudnnCreataAttnDescriptor(ptr)
        cudnnSetAttnDescriptor(ptr[1], args...)
        mha = cudnnAttnDescriptor(ptr[1])
        finalizer(x->cudnnDestroyAttnDescriptor(x.ptr), mha)
        return mha
    end
end


function cudnnMultiHeadAttnForward(x;
    mode, nHeads, DT(dataType), DT(computePrec), mathType, DD(attnDropout), DD(postDropout), qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize,
             attnMode::Unsigned = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | CUDNN_ATTN_DISABLE_PROJ_BIASES,
             nHeads::Integer,
             smScaler::Real = 1,
             dataType::DataType,
             computePrec::DataType = dataType, # There doesn't seem to be any other option in cudnn 8.0.2 docs
             mathType::cudnnMathType_t = (dataType === Float16 ? CUDNN_TENSOR_OP_MATH :
                                          dataType === Float32 ? CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION :
                                          CUDNN_DEFAULT_MATH),
             attnDropout::Real = 0,
    postDropout::Real = 0,
                              )
    cudnnMultiHeadAttnForward()
    return
end

function multiHeadAttnBackwardData
end

function multiHeadAttnBackwardWeights
end

@primitive1 multiHeadAttnForward(x; o...),dy,y  multiHeadAttnBackwardWeights(x,y,dy; o...)  multiHeadAttnBackwardData(x,y,dy; o...)
@primitive1 multiHeadAttnBackwardData(x,y...;o...)  throw(MethodError(back,multiHeadAttnBackwardData))
@primitive1 multiHeadAttnBackwardWeights(x,y...;o...)  throw(MethodError(back,multiHeadAttnBackwardWeights))
