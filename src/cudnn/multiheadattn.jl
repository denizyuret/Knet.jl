using Knet.KnetArrays: DevArray
# @primitive1 is too inefficient for functions with many args, we will define gradients manually:
using AutoGrad: forw, Value, Arg, recording
import AutoGrad: back
AutoGrad.@zerograd sizeof(x)  # TODO: remove after fixing in AutoGrad

using CUDA.CUDNN: 
   #cudnnMultiHeadAttnForward,
    cudnnMultiHeadAttnBackwardData,
    cudnnMultiHeadAttnBackwardWeights,
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
        CUDNN_FMA_MATH,                        # 3,
    cudnnWgradMode_t,
        CUDNN_WGRAD_MODE_ADD,  # 0,
        CUDNN_WGRAD_MODE_SET,  # 1,
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
    


cudnnMultiHeadAttnForward(w,q,k,v; o...)               = cudnnMultiHeadAttnForwardWithDefaults(w,q,k,v; o...)
cudnnMultiHeadAttnForward(w,q,k,v,attnDesc; o...)      = cudnnMultiHeadAttnForwardWithDefaults(w,q,k,v; attnDesc, o...)
cudnnMultiHeadAttnForward!(out,w,q,k,v; o...)          = cudnnMultiHeadAttnForwardWithDefaults(w,q,k,v; out, o...)
cudnnMultiHeadAttnForward!(out,w,q,k,v,attnDesc; o...) = cudnnMultiHeadAttnForwardWithDefaults(w,q,k,v; out, attnDesc, o...)


function cudnnMultiHeadAttnForwardWithDefaults(
    weights, queries, keys, values;

    # Buffers for gradients and tensor sizes
    dweights = (recording() && weights !== nothing ? similar(weights) : nothing),
    dqueries = (recording() ? similar(queries) : nothing),
    dkeys    = (recording() ? similar(keys) : nothing),
    dvalues  = (recording() ? similar(values) : nothing),
    _qdims::Vector{Cint} = sdim4(size(queries)),
    _kdims::Vector{Cint} = sdim4(size(keys)),

    # attnDesc parameters
    attnMode::Unsigned = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | CUDNN_ATTN_DISABLE_PROJ_BIASES |> Unsigned,
    nHeads::Integer = 2,
    smScaler::Real = 1,
    # dataType::DataType = eltype(queries),
    # computePrec::DataType = eltype(queries),  ## Only option according to 8.0.2
    # mathType::cudnnMathType_t = cudnnMultiHeadAttnMathType(eltype(queries)),  ## Always pick tensor math if available
    # attnDropout::Real = 0, ## The dropout option is currently not supported by the multi-head attention API
    # postDropout::Real = 0,
    qProjSize::Integer = 0, # Use zero to disable the corresponding projection
    kProjSize::Integer = 0,
    vProjSize::Integer = 0,
    oProjSize::Integer = 0,
    qoMaxSeqLength::Integer = _qdims[1],
    kvMaxSeqLength::Integer = _kdims[1],
    maxBatchSize::Integer = _qdims[2],
    maxBeamSize::Integer = _qdims[3],

    # initialize attnDesc
    attnDesc::cudnnAttnDescriptor = cudnnAttnDescriptor(
        Cuint(attnMode),
        Cint(nHeads),
        Cdouble(smScaler),
        DT(eltype(queries)),    # dataType
        DT(eltype(queries)),    # computePrec
        cudnnMultiHeadAttnMathType(eltype(queries)), # mathType
        C_NULL,  # attnDropout
        C_NULL,  # postDropout
        Cint(size(queries,1)),  # qSize
        Cint(size(keys,1)),     # kSize
        Cint(size(values,1)),   # vSize
        Cint(qProjSize),
        Cint(kProjSize),
        Cint(vProjSize),
        Cint(oProjSize),
        Cint(qoMaxSeqLength),
        Cint(kvMaxSeqLength),
        Cint(maxBatchSize),
        Cint(maxBeamSize)
    ),

    # forw parameters
    out=nothing,
    residuals = nothing,
    currIdx::Integer = -1,
    loWinIdx::Array{Cint} = fill(Cint(0), qoMaxSeqLength),
    hiWinIdx::Array{Cint} = fill(Cint(kvMaxSeqLength), qoMaxSeqLength),
    workSpace::Union{DevArray,Nothing}    = nothing, 
    reserveSpace::Union{DevArray,Nothing} = nothing,
    seqLengthsQO::Vector{<:Integer} = fill(_qdims[1], _qdims[2]*_qdims[3]),
    seqLengthsKV::Vector{<:Integer} = fill(_kdims[1], _kdims[2]*_kdims[3]),
    devSeqLengthsQO::DevArray{Cint,1} = convert(CuArray{Cint}, seqLengthsQO),
    devSeqLengthsKV::DevArray{Cint,1} = convert(CuArray{Cint}, seqLengthsKV),
    axes::Vector{cudnnSeqDataAxis_t} = cudnnSeqDataDefaultAxes,
    qDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(queries; axes, seqLengthArray=seqLengthsQO),
    kDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(keys;    axes, seqLengthArray=seqLengthsKV),
    vDesc::cudnnSeqDataDescriptor = cudnnSeqDataDescriptor(values;  axes, seqLengthArray=seqLengthsKV),
    oDesc::Union{cudnnSeqDataDescriptor,Nothing} = cudnnSeqDataDescriptor(out; axes, seqLengthArray=seqLengthsQO),
)
    (wSize, wsSize, rsSize) = cudnnMultiHeadAttnBuffers(attnDesc)
    @assert sizeof(weights) == wSize  "weights should be $wSize bytes."
    qSize = (qProjSize > 0 ? qProjSize : size(queries,1))
    kSize = (kProjSize > 0 ? kProjSize : size(keys,1))
    @assert kSize == qSize  "key size $kSize does not match query size $qSize."
    vSize = (vProjSize > 0 ? vProjSize : size(values,1))
    @assert size(keys)[2:end] == size(values)[2:end]  "key tensor $(size(keys)) does not match value tensor $(size(values))"
    oSize = (oProjSize > 0 ? oProjSize : nHeads * vSize)
    oDims = (oSize, size(queries)[2:end]...)
    @assert residuals === nothing || size(residuals) == oDims  "residual size should be $(oDims)"
    out === nothing ? out = similar(values, oDims) : @assert size(out) == oDims  "output size should be $(oDims)"
    if oDesc === nothing; oDesc = cudnnSeqDataDescriptor(out, seqLengthArray=seqLengthsQO); end

    if recording()
        @assert weights === nothing || size(dweights)==size(weights)
        @assert size(dqueries) == size(queries)
        @assert size(dkeys) == size(keys)
        @assert size(dvalues) == size(values)
    end

    @assert (attnMode & CUDNN_ATTN_ENABLE_PROJ_BIASES == 0) "The CUDNN_ATTN_ENABLE_PROJ_BIASES option is not supported in the multi-head attention gradient functions."
    @assert smScaler >= 0  "The user can set smScaler to any positive floating-point value or even zero."
    # These options are not really options in 8.0.2:
    # @assert computePrec === dataType  "Only computePrec === dataType supported as of cuDNN 8.0.2."
    # @assert(((mathType === CUDNN_DEFAULT_MATH) ||
    #          (mathType === CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION && dataType in (Float16, Float32)) ||
    #          (mathType === CUDNN_TENSOR_OP_MATH && dataType === Float16)),
    #         "Unsupported mathType $mathType for dataType $dataType.")
    # @assert (attnDropout == postDropout == 0) "The dropout option is not supported by the multi-head attention API as of cuDNN 8.0.2."
    @assert qoMaxSeqLength >= _qdims[1]
    @assert kvMaxSeqLength >= _kdims[1]
    @assert maxBatchSize >= _qdims[2]
    @assert maxBeamSize >= _qdims[3]
    @assert _kdims[2] == _qdims[2]
    @assert _kdims[3] == (((attnMode & CUDNN_ATTN_QUERYMAP_ONE_TO_ONE) > 0) ? _qdims[3] : 1)

    if wsSize > 0 && workSpace === nothing; workSpace = cudnnMultiHeadAttnBuffer(wsSize); end
    if rsSize > 0 && reserveSpace === nothing; reserveSpace = cudnnMultiHeadAttnBuffer(rsSize); end
    @assert sizeof(workSpace) >= wsSize  "worksSpace should be at least $wsSize bytes"
    @assert sizeof(reserveSpace) >= rsSize  "reserveSpace should be at least $rsSize bytes"

    forw(cudnnMultiHeadAttnForwardAutoGrad,
         weights, queries, keys, values, residuals;
         dweights, dqueries, dkeys, dvalues, # dresiduals is equal to dout
         attnDesc, currIdx, loWinIdx, hiWinIdx,
         devSeqLengthsQO, devSeqLengthsKV,
         qDesc, kDesc, vDesc, oDesc,
         out, workSpace, reserveSpace)
end


function cudnnMultiHeadAttnForwardAutoGrad(
    weights, queries, keys, values, residuals;
    dweights, dqueries, dkeys, dvalues,
    attnDesc, currIdx, loWinIdx, hiWinIdx,
    devSeqLengthsQO, devSeqLengthsKV,
    qDesc, kDesc, vDesc, oDesc,
    out, workSpace, reserveSpace
)
    CUDA.CUDNN.cudnnMultiHeadAttnForward(
        handle(), attnDesc, currIdx,
        loWinIdx, hiWinIdx,
        devSeqLengthsQO, devSeqLengthsKV,
        qDesc, queries, cu_null(residuals),
        kDesc, keys,
        vDesc, values,
        oDesc, out,
        sizeof(weights), cu_null(weights),
        sizeof(workSpace), cu_null(workSpace),
        sizeof(reserveSpace), cu_null(reserveSpace)
    )
    return out
end


# We do all the work during the backward pass for the first arg
function back(
    ::typeof(cudnnMultiHeadAttnForwardAutoGrad), ::Type{Arg{1}}, dout, _out, 
    weights, queries, keys, values, residuals;
    dweights, dqueries, dkeys, dvalues,
    attnDesc, currIdx, loWinIdx, hiWinIdx,
    devSeqLengthsQO, devSeqLengthsKV,
    qDesc, kDesc, vDesc, oDesc,
    out, workSpace, reserveSpace)

    (weights, queries, keys, values, residuals) = value.((weights, queries, keys, values, residuals))

    cudnnMultiHeadAttnBackwardData(
        handle(), attnDesc,
        loWinIdx, hiWinIdx,
        devSeqLengthsQO, devSeqLengthsKV,
        oDesc, dout,
        qDesc, dqueries, queries,
        kDesc, dkeys, keys,
        vDesc, dvalues, values,
        sizeof(weights), cu_null(weights),
        sizeof(workSpace), cu_null(workSpace),
        sizeof(reserveSpace), cu_null(reserveSpace))

     weights !== nothing && cudnnMultiHeadAttnBackwardWeights(
         handle(), attnDesc,
         CUDNN_WGRAD_MODE_SET,
         qDesc, queries,
         kDesc, keys,
         vDesc, values,
         oDesc, dout,
         sizeof(weights), cu_null(weights), cu_null(dweights),
         sizeof(workSpace), cu_null(workSpace),
         sizeof(reserveSpace), cu_null(reserveSpace))

    return dweights
end

# The backward pass for the other args only return already computed gradients
back(::typeof(cudnnMultiHeadAttnForwardAutoGrad), ::Type{Arg{2}}, x...; dqueries, o...) = dqueries
back(::typeof(cudnnMultiHeadAttnForwardAutoGrad), ::Type{Arg{3}}, x...; dkeys, o...) = dkeys
back(::typeof(cudnnMultiHeadAttnForwardAutoGrad), ::Type{Arg{4}}, x...; dvalues, o...) = dvalues
back(::typeof(cudnnMultiHeadAttnForwardAutoGrad), ::Type{Arg{5}}, dout, _out, weights, queries, keys, values, residuals; o...) =
    (residuals === nothing ? nothing : dout)


# Residuals: The cudnnMultiHeadAttnBackwardData() function does not output partial
# derivatives for residual connections because this result is equal to dout . If the
# multi-head attention model enables residual connections sourced directly from Q, then the
# dout tensor needs to be added to dqueries to obtain the correct result of the latter. This
# operation is demonstrated in the cuDNN multiHeadAttention sample code.
# AutoGrad automatically does the addition.
# What if q = LayerNorm(r)? In my tests dr=dout and dq=0. TODO: check this further.



cudnnMultiHeadAttnMathType(::Type) = CUDNN_DEFAULT_MATH
cudnnMultiHeadAttnMathType(::Type{Float16}) = CUDNN_TENSOR_OP_MATH
cudnnMultiHeadAttnMathType(::Type{Float32}) = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION

function cudnnMultiHeadAttnBuffers(attnDesc::cudnnAttnDescriptor)
    weightSize, workSpaceSize, reserveSpaceSize = ntuple(i->Csize_t[0], 3)
    cudnnGetMultiHeadAttnBuffers(handle(), attnDesc, weightSize, workSpaceSize, recording() ? reserveSpaceSize : C_NULL)
    return (weightSize[1], workSpaceSize[1], reserveSpaceSize[1])
end

function cudnnMultiHeadAttnBuffer(bytes::Integer)
    return CuArray{Int128}(undef, (bytes-1)÷sizeof(Int128)+1)
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

function cudnnSeqDataDescriptor(
    array; 
    dataType::DataType = eltype(array),
    nbDims::Integer = 4, # cudnn-doc: The number of active dimensions in the dimA[] and axes[] arrays is defined by the nbDims argument. Currently, the value of this argument should be four. The actual size of the dimA[] and axes[] arrays should be declared using the CUDNN_SEQDATA_DIM_COUNT macro.
    dimA::Vector{<:Integer} = sdim4(size(array)),
    axes::Vector{cudnnSeqDataAxis_t} = cudnnSeqDataDefaultAxes,
    seqLengthArray::Vector{<:Integer} = fill(dimA[1], dimA[2]*dimA[3]), # cudnn-doc: The seqLengthArray[] must specify all sequence lengths in the container so the total size of this array should be dimA[CUDNN_SEQDATA_BATCH_DIM] * dimA[CUDNN_SEQDATA_BEAM_DIM].
    seqLengthArraySize::Integer = Csize_t(length(seqLengthArray)),
    paddingFill::Ptr{Cvoid} = C_NULL, # cudnn-doc: Currently, the only supported value for paddingFill is NULL which means this option should be ignored.
)
    cudnnSeqDataDescriptor(DT(dataType), Cint(nbDims), convert(Vector{Cint}, dimA), axes, 
                           convert(Csize_t,seqLengthArraySize), convert(Vector{Cint}, seqLengthArray), paddingFill)
end

cudnnSeqDataDescriptor(::Nothing; o...) = nothing
