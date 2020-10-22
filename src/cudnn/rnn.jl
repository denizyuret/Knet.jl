using AutoGrad: AutoGrad, @primitive1, value, recording

using CUDA.CUDNN:
#cudnnRNNForward,
cudnnRNNDescriptor_t,
cudnnSetRNNDescriptor_v8,
cudnnRNNAlgo_t,
    CUDNN_RNN_ALGO_STANDARD,        # 0, robust performance across a wide range of network parameters
    CUDNN_RNN_ALGO_PERSIST_STATIC,  # 1, fast when the first dimension of the input tensor is small (meaning, a small minibatch), cc>=6.0
    CUDNN_RNN_ALGO_PERSIST_DYNAMIC, # 2, similar to static, optimize using the specific parameters of the network and active GPU, cc>=6.0
    CUDNN_RNN_ALGO_COUNT,           # 3
cudnnRNNMode_t,
    CUDNN_RNN_RELU, # 0, /* basic RNN cell type with ReLu activation */
    CUDNN_RNN_TANH, # 1, /* basic RNN cell type with tanh activation */
    CUDNN_LSTM,     # 2, /* LSTM with optional recurrent projection and clipping */
    CUDNN_GRU,      # 3, /* Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1); */
cudnnRNNBiasMode_t,
    CUDNN_RNN_NO_BIAS,         # 0, /* rnn cell formulas do not use biases */
    CUDNN_RNN_SINGLE_INP_BIAS, # 1, /* rnn cell formulas use one input bias in input GEMM */
    CUDNN_RNN_DOUBLE_BIAS,     # 2, /* default, rnn cell formulas use two bias vectors */
    CUDNN_RNN_SINGLE_REC_BIAS, # 3  /* rnn cell formulas use one recurrent bias in recurrent GEMM */
cudnnDirectionMode_t,
    CUDNN_UNIDIRECTIONAL, # 0, /* single direction network */
    CUDNN_BIDIRECTIONAL,  # 1, /* output concatination at each layer */
cudnnRNNInputMode_t,
    CUDNN_LINEAR_INPUT, # 0, /* adjustable weight matrix in first layer input GEMM */
    CUDNN_SKIP_INPUT,   # 1, /* fixed identity matrix in the first layer input GEMM */
cudnnMathType_t,
    CUDNN_DEFAULT_MATH,                    # 0,
    CUDNN_TENSOR_OP_MATH,                  # 1,
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION, # 2,
    CUDNN_FMA_MATH,                        # 3,
#/* For auxFlags in cudnnSetRNNDescriptor_v8() and cudnnSetRNNPaddingMode() */
    CUDNN_RNN_PADDED_IO_DISABLED, # 0
    CUDNN_RNN_PADDED_IO_ENABLED,  # (1U << 0)
cudnnForwardMode_t,
cudnnRNNDataDescriptor_t,
cudnnSetRNNDataDescriptor,
cudnnRNNDataLayout_t,
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,   # 0, /* padded, outer stride from one time-step to the next */
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,     # 1, /* sequence length sorted and packed as in basic RNN api */
    CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED, # 2, /* padded, outer stride from one batch to the next */
cudnnTensorDescriptor_t,
handle


cudnnRNNForward(w, x, hx, cx; o...)              = cudnnRNNForwardWithDefaults(w, x, hx, cx; o...)
cudnnRNNForward(w, x, hx, cx, rnnDesc; o...)     = cudnnRNNForwardWithDefaults(w, x, hx, cx; rnnDesc, o...)
cudnnRNNForward!(y, w, x, hx, cx; o...)          = cudnnRNNForwardWithDefaults(w, x, hx, cx; y, o...)
cudnnRNNForward!(y, w, x, hx, cx, rnnDesc; o...) = cudnnRNNForwardWithDefaults(w, x, hx, cx; y, rnnDesc, o...)


function cudnnRNNForwardWithDefaults(
    w, x, hx, cx;
    # rnnDescriptor parameters
    algo::cudnnRNNAlgo_t = CUDNN_RNN_ALGO_STANDARD,
    cellMode::cudnnRNNMode_t = CUDNN_LSTM,
    biasMode::cudnnRNNBiasMode_t = CUDNN_RNN_DOUBLE_BIAS,
    dirMode::cudnnDirectionMode_t = CUDNN_UNIDIRECTIONAL,
    inputMode::cudnnRNNInputMode_t = CUDNN_LINEAR_INPUT,
    dataType::DataType = eltype(x),
    mathPrec::DataType = dataType, # has to match dataType with one exception dt=Float16 => mp=Float16|Float32
    mathType::cudnnMathType_t = cudnnRNNMathType(dataType),
    inputSize::Integer = 0,
    hiddenSize::Integer = 0,
    projSize::Integer = hiddenSize,
    numLayers::Integer = 1,
    dropout::Real = 0,
    auxFlags::Integer = CUDNN_RNN_PADDED_IO_DISABLED, # When the padded I/O is enabled, layouts CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED and CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED are permitted in RNN data descriptors.
    # rnnDescriptor
    rnnDesc::cudnnRNNDescriptor = cudnnRNNDescriptor(algo, cellMode, biasMode, dirMode, inputMode, cudnnDataType(dataType), cudnnDataType(mathPrec), mathType, Int32(inputSize), Int32(hiddenSize), Int32(projSize), Int32(numLayers), cudnnDropoutDescriptor(dropout), UInt32(auxFlags)),
    # rnnData parameters
    layout::cudnnRNNDataLayout_t = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, # padded [X,B,T] array
    maxSeqLength::Integer = size(x,3),
    batchSize::Integer = size(x,2),
    xVectorSize::Integer = inputSize,
    yVectorSize::Integer = projSize * (dirMode === CUDNN_UNIDIRECTIONAL ? 1 : 2),
    hVectorSize::Integer = projSize,
    cVectorSize::Integer = hiddenSize,
    seqLengthArray::Vector{Cint} = fill(Cint(size(x,3)), size(x,2)),
    paddingFill::Ptr{Cvoid} = C_NULL,
    # rnnData descriptors
    xDesc::cudnnRNNDataDescriptor = cudnnRNNDataDescriptor(cudnnDataType(dataType), layout, Cint(maxSeqLength), Cint(batchSize), Cint(xVectorSize), seqLengthArray, paddingFill),
    yDesc::cudnnRNNDataDescriptor = cudnnRNNDataDescriptor(cudnnDataType(dataType), layout, Cint(maxSeqLength), Cint(batchSize), Cint(yVectorSize), seqLengthArray, paddingFill),
    hDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(CUDNN_TENSOR_NCHW, cudnnDataType(dataType), Cint(3), Cint[numLayers * (dirMode === CUDNN_UNIDIRECTIONAL ? 1 : 2), batchSize, hVectorSize]),
    cDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(CUDNN_TENSOR_NCHW, cudnnDataType(dataType), Cint(3), Cint[numLayers * (dirMode === CUDNN_UNIDIRECTIONAL ? 1 : 2), batchSize, cVectorSize]),
    # rnnForward parameters
    y = similar(x, yVectorSize, size(x)[2:end]...),
    hy = nothing,
    cy = nothing,
    fwdMode::cudnnForwardMode_t = recording() ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE,
    devSeqLengths::DevArray{Cint,1} = CuArray(seqLengthArray),
    weightSpace = cudnnRNNWeightSpace(),
    workSpace = cudnnRNNWorkSpace(),
    reserveSpace = cudnnRNNReserveSpace()
)
    cudnnRNNForwardAutoGrad(w, x, hx, cx; rnnDesc, fwdMode, devSeqLengths, xDesc, yDesc, y, hDesc, hy, cDesc, cy, weightSpace, workSpace, reserveSpace)
end


function cudnnRNNForwardAutoGrad(w, x, hx, cx; rnnDesc, fwdMode, devSeqLengths, xDesc, yDesc, y, hDesc, hy, cDesc, cy, weightSpace, workSpace, reserveSpace)
    CUDA.CUDNN.cudnnRNNForward(handle(), rnnDesc, fwdMode, devSeqLengths, xDesc, x, yDesc, y, hDesc, cu_null(hx), cu_null(hy), cDesc, cu_null(cx), cu_null(cy), sizeof(weightSpace), cu_null(weightSpace), sizeof(workSpace), cu_null(workSpace), sizeof(reserveSpace), cu_null(reserveSpace))
    return (y, hy, cy)
end


# we may have to do this manually like in multiheadattn
@primitive1((cudnnRNNForwardAutoGrad(w, x, hx, cx; rnnDesc, fwdMode, devSeqLengths, xDesc, yDesc, y, hDesc, hy, cDesc, cy, weightSpace, workSpace, reserveSpace),
             _dy,_y),
            (cudnnRNNBackwardWeights_v8()),
            (cudnnRNNBackwardData_v8()),
            )


cudnnRNNMathType(::Type{Float16})=CUDNN_TENSOR_OP_MATH
cudnnRNNMathType(::Type{Float32})=CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
cudnnRNNMathType(::Type{Float64})=CUDNN_DEFAULT_MATH

# todo workspace etc.
