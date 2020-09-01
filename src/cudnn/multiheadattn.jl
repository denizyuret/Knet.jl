import Base: unsafe_convert
using Knet.KnetArrays: DevArray
using AutoGrad: AutoGrad, @primitive1, recording
using CUDA: CU_NULL

using CUDA.CUDNN: 
   #cudnnMultiHeadAttnForward,
   #cudnnMultiHeadAttnBackwardData,
   #cudnnMultiHeadAttnBackwardWeights,
    cudnnGetMultiHeadAttnBuffers,
    cudnnGetMultiHeadAttnWeights,
    cudnnAttnDescriptor_t,
        cudnnCreateAttnDescriptor,
        cudnnDestroyAttnDescriptor,
        cudnnSetAttnDescriptor,
        cudnnGetAttnDescriptor,
        cudnnDataType_t,
        cudnnDropoutDescriptor_t,
    cudnnAttnQueryMap_t,
        CUDNN_ATTN_QUERYMAP_ALL_TO_ONE, # 0         /* multiple Q-s map to a single (K,V) set when beam size > 1, beam sizes for (K,V) = 1 */
        CUDNN_ATTN_QUERYMAP_ONE_TO_ONE, # (1U << 0) /* multiple Q-s map to multiple (K,V) sets when beam size > 1, beam sizes for (K,V) = beam size for (Q) */
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
       #CUDNN_FMA_MATH,                        # 3,
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
        CUDNN_SEQDATA_DIM_COUNT, # 4
    handle
    

## cudnnSeqDataDescriptor:

mutable struct cudnnSeqDataDescriptor; ptr::cudnnSeqDataDescriptor_t; end

unsafe_convert(::Type{<:Ptr}, mha::cudnnSeqDataDescriptor)=mha.ptr

const cudnnSeqDataDescriptorCache = Dict{Tuple,cudnnSeqDataDescriptor}()

function cudnnSeqDataDescriptor(args...)
    get!(cudnnSeqDataDescriptorCache, args) do
        ptr = cudnnSeqDataDescriptor_t[C_NULL]
        cudnnCreateSeqDataDescriptor(ptr)
        cudnnSetSeqDataDescriptor(ptr[1], args...)
        d = cudnnSeqDataDescriptor(ptr[1])
        finalizer(x->cudnnDestroySeqDataDescriptor(x.ptr), d)
        return d
    end
end

# Note that axes and dimA are reversed in cudnn relative to Julia size(), so VECT is always Julia dim=1.
const cudnnSeqDataDefaultAxes = cudnnSeqDataAxis_t[
    CUDNN_SEQDATA_TIME_DIM,
    CUDNN_SEQDATA_BATCH_DIM,
    CUDNN_SEQDATA_BEAM_DIM,
    CUDNN_SEQDATA_VECT_DIM
]

# For tensors with less than 4 dims we assume size=(VECT,BATCH,TIME) Julia order with BEAM=1.
sdim4(s::Dims{0}) = Cint[1,1,1,1]
sdim4(s::Dims{1}) = Cint[1,1,1,s[1]] # assume single dim is VECT
sdim4(s::Dims{2}) = Cint[1,s[2],1,s[1]] # assume two dims is VECT,BATCH
sdim4(s::Dims{3}) = Cint[s[3],s[2],1,s[1]] # assume three dims is VECT,BATCH,TIME
sdim4(s::Dims{4}) = Cint[s[4],s[3],s[2],s[1]] # assume four dims is VECT,BEAM,BATCH,TIME
sdim4(s::Dims{N}) where N = error("cudnnSeqDataDescriptor only supports up to 4 dims.")

function cudnnSeqDataDescriptor(a)
    dataType = DT(eltype(a))
    nbDims = Cint(4) # cudnn-doc: The number of active dimensions in the dimA[] and axes[] arrays is defined by the nbDims argument. Currently, the value of this argument should be four. The actual size of the dimA[] and axes[] arrays should be declared using the CUDNN_SEQDATA_DIM_COUNT macro.
    dimA = sdim4(size(a))
    axes = cudnnSeqDataDefaultAxes
    seqLengthArraySize = Csize_t(dimA[2]*dimA[3])
    seqLengthArray = fill(dimA[1], seqLengthArraySize) # cudnn-doc: The seqLengthArray[] must specify all sequence lengths in the container so the total size of this array should be dimA[CUDNN_SEQDATA_BATCH_DIM] * dimA[CUDNN_SEQDATA_BEAM_DIM].
    paddingFill = C_NULL # cudnn-doc: Currently, the only supported value for paddingFill is NULL which means this option should be ignored.
    cudnnSeqDataDescriptor(dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray, paddingFill)
end


## cudnnAttnDescriptor:

mutable struct cudnnAttnDescriptor; ptr::cudnnAttnDescriptor_t; end

unsafe_convert(::Type{<:Ptr}, mha::cudnnAttnDescriptor)=mha.ptr

const cudnnAttnDescriptorCache = Dict{Tuple,cudnnAttnDescriptor}()

function cudnnAttnDescriptor(args...)
    get!(cudnnAttnDescriptorCache, args) do
        ptr = cudnnAttnDescriptor_t[C_NULL]
        cudnnCreateAttnDescriptor(ptr)
        cudnnSetAttnDescriptor(ptr[1], args...)
        mha = cudnnAttnDescriptor(ptr[1])
        finalizer(x->cudnnDestroyAttnDescriptor(x.ptr), mha)
        return mha
    end
end


## cudnnMultiHeadAttnForward:

function cudnnMultiHeadAttnForward(
    weights::R, queries::R, keys::R, values::R, out::R = similar(values); # TODO: use correct weight/output size, should residuals be here? do all arrays have the same ndims?

    attnMode::Unsigned = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | CUDNN_ATTN_DISABLE_PROJ_BIASES |> Unsigned, # The CUDNN_ATTN_ENABLE_PROJ_BIASES option is not supported in the multi-head attention gradient functions.
    nHeads::Integer = 2,
    smScaler::Real = 1,
    dataType::DataType = T,
    computePrec::DataType = dataType, # There doesn't seem to be any other option in cudnn 8.0.2 docs
    mathType::cudnnMathType_t = cudnnMultiHeadAttnMathType(dataType),
    attnDropout::Real = 0,
    postDropout::Real = 0,
    #qSize::Integer = size(queries, 1), #The first dim of Q,K,V is always the vector dimension, the other 3 can be any permutation of beam, batch, and time
    #kSize::Integer = size(keys, 1),    #So the these are not user settable:
    #vSize::Integer = size(values, 1),
    qProjSize::Integer = 0, # Use zero to disable the corresponding projection
    kProjSize::Integer = 0,
    vProjSize::Integer = 0,
    oProjSize::Integer = 0,
    qoMaxSeqLength::Integer = 128,
    kvMaxSeqLength::Integer = qoMaxSeqLength,
    maxBatchSize::Integer = 32,
    maxBeamSize::Integer = 1,

    attnDesc::cudnnAttnDescriptor = cudnnAttnDescriptor(
        Cuint(attnMode),
        Cint(nHeads),
        Cdouble(smScaler),
        DT(dataType),
        DT(computePrec),
        mathType,
        attnDropout == 0 ? C_NULL : error("The dropout option is currently not supported by the multi-head attention API"), # cudnnDropoutDescriptor(attnDropout), # TODO: DropoutDescriptor is a bad thing to has when this option is available 
        postDropout == 0 ? C_NULL : error("The dropout option is currently not supported by the multi-head attention API"), # cudnnDropoutDescriptor(postDropout),
        Cint(size(queries,1)),
        Cint(size(keys,1)),
        Cint(size(values,1)),
        Cint(qProjSize),
        Cint(kProjSize),
        Cint(vProjSize),
        Cint(oProjSize),
        Cint(qoMaxSeqLength),
        Cint(kvMaxSeqLength),
        Cint(maxBatchSize),
        Cint(maxBeamSize)
    ),
    currIdx::Integer = -1,
    loWinIdx::Array{Cint} = fill(Cint(0), qoMaxSeqLength),
    hiWinIdx::Array{Cint} = fill(typemax(Cint), qoMaxSeqLength),
    residuals::Union{R,Nothing} = nothing, # TODO: make sure gradients pass through residuals correctly if used
    workSpace::DevArray = cudnnMultiHeadAttnWorkSpace(attnDesc),
    reserveSpace::Union{DevArray,Nothing} = (recording() ? cudnnMultiHeadAttnReserveSpace(attnDesc) : nothing),
    qDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(queries),
    kDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(keys),
    vDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(values),
    oDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(out),
    devSeqLengthsQO::DevArray{Cint} = cudnnSeqLengths(qDesc, maxBatchSize*maxBeamSize+1),
    devSeqLengthsKV::DevArray{Cint} = cudnnSeqLengths(kDesc, maxBatchSize*maxBeamSize+1)
) where {T,R<:DevArray{T}}
    global _ad = attnDesc
    cu_null(x) = (x === nothing ? CU_NULL : x)
    CUDA.CUDNN.cudnnMultiHeadAttnForward(handle(), attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, queries, cu_null(residuals), kDesc, keys, vDesc, values, oDesc, out, sizeof(weights), weights, sizeof(workSpace), workSpace, sizeof(reserveSpace), cu_null(reserveSpace))
    return out
end

function cudnnSeqLengths(d::cudnnSeqDataDescriptor, seqLengthSizeRequested=128)
    seqLengthArray = Array{Cint}(undef, seqLengthSizeRequested)
    seqLengthArraySize = Csize_t[0]
    cudnnGetSeqDataDescriptor(d, C_NULL, C_NULL, 0, C_NULL, C_NULL, seqLengthArraySize, seqLengthSizeRequested, seqLengthArray, C_NULL)
    if seqLengthArraySize[1] < seqLengthSizeRequested 
        return CuArray(seqLengthArray[1:seqLengthArraySize[1]])
    else
        return cudnnSeqLengths(d, 2*seqLengthSizeRequested)
    end
end

cudnnMultiHeadAttnMathType(::Type) = CUDNN_DEFAULT_MATH
cudnnMultiHeadAttnMathType(::Type{Float16}) = CUDNN_TENSOR_OP_MATH
cudnnMultiHeadAttnMathType(::Type{Float32}) = CUDNN_DEFAULT_MATH # CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION

function cudnnMultiHeadAttnReserveSpace(attnDesc::cudnnAttnDescriptor)
    weightSize, workSpaceSize, reserveSpaceSize = ntuple(i->Csize_t[0], 3)
    cudnnGetMultiHeadAttnBuffers(handle(), attnDesc, weightSize, workSpaceSize, reserveSpaceSize)
    return CuArray{Int}(undef, (reserveSpaceSize[1]-1)÷sizeof(Int)+1)
end

function cudnnMultiHeadAttnWorkSpace(attnDesc::cudnnAttnDescriptor)
    weightSize, workSpaceSize, reserveSpaceSize = ntuple(i->Csize_t[0], 3)
    cudnnGetMultiHeadAttnBuffers(handle(), attnDesc, weightSize, workSpaceSize, reserveSpaceSize)
    return CuArray{Int}(undef, (workSpaceSize[1]-1)÷sizeof(Int)+1)
end

@primitive1((multiHeadAttnForward(x; o...),dy,y),  
            multiHeadAttnBackwardData(x,y,dy; o...),
            multiHeadAttnBackwardWeights(x,y,dy; o...))
@primitive1 cudnnMultiHeadAttnBackwardData(x,y...;o...)     throw(MethodError(back,cudnnMultiHeadAttnBackwardData))
@primitive1 cudnnMultiHeadAttnBackwardWeights(x,y...;o...)  throw(MethodError(back,cudnnMultiHeadAttnBackwardWeights))

# See the following for some of the default values:
#
# * https://github.com/google-research/bert/blob/master/README.md
# * https://arxiv.org/abs/1908.08962
# * https://arxiv.org/abs/1706.03762
# * https://huggingface.co/bert-base-uncased/models
#
# bert-large: nHeads=16, nLayers=24, hiddenSize=1024, intermSize=4096, maxSeqLen=512, attnDropout=postDropout=0.1, init=0.02
# bert-base:  nHeads=12, nLayers=12, hiddenSize=768,  intermSize=3072, maxSeqLen=512, attnDropout=postDropout=0.1, init=0.02
# bert-medium:nHeads=8,  nLayers=8,  hiddenSize=512,  intermSize=2048, maxSeqLen=512, attnDropout=postDropout=0.1, init=0.02
# bert-small: nHeads=8,  nLayers=4,  hiddenSize=512,  intermSize=2048, maxSeqLen=512, attnDropout=postDropout=0.1, init=0.02
# bert-mini:  nHeads=4,  nLayers=4,  hiddenSize=256,  intermSize=1024, maxSeqLen=512, attnDropout=postDropout=0.1, init=0.02
# bert-tiny:  nHeads=2,  nLayers=2,  hiddenSize=128,  intermSize=512,  maxSeqLen=512, attnDropout=postDropout=0.1, init=0.02
# vasw-base:  nHeads=8,  nLayers=6,  hiddenSize=512,  intermSize=2048
# vasw-big:   nHeads=16, nLayers=6,  hiddenSize=1024, intermSize=4096

