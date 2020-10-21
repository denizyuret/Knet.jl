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
/* For auxFlags in cudnnSetRNNDescriptor_v8() and cudnnSetRNNPaddingMode() */
    CUDNN_RNN_PADDED_IO_DISABLED, # 0
    CUDNN_RNN_PADDED_IO_ENABLED,  # (1U << 0)
cudnnForwardMode_t,
cudnnRNNDataDescriptor_t,
cudnnSetRNNDataDescriptor,
cudnnTensorDescriptor_t,
handle



function cudnnRNNForwardWithDefaults(
    w, x;
    # rnnDescriptor parameters
    algo::cudnnRNNAlgo_t = CUDNN_RNN_ALGO_STANDARD,
    cellMode::cudnnRNNMode_t = CUDNN_LSTM,
    biasMode::cudnnRNNBiasMode_t = CUDNN_RNN_DOUBLE_BIAS,
    dirMode::cudnnDirectionMode_t = CUDNN_UNIDIRECTIONAL,
    inputMode::cudnnRNNInputMode_t = CUDNN_LINEAR_INPUT,
    dataType::DataType = eltype(w),
    mathPrec::DataType = dataType, # has to match dataType except when dt=Float16, mp can also be Float32
    mathType::cudnnMathType_t = cudnnRNNMathType(dataType),
    inputSize::Integer = 0,
    hiddenSize::Integer = 0,
    projSize::Integer = hiddenSize,
    numLayers::Integer = 1,
    dropout::Real = 0,
    auxFlags::Integer = CUDNN_RNN_PADDED_IO_DISABLED, # When the padded I/O is enabled, layouts CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED and CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED are permitted in RNN data descriptors.
    # rnnDescriptor
    rnnDesc = cudnnRNNDescriptor(algo, cellMode, biasMode, dirMode, inputMode, cudnnDataType(dataType), cudnnDataType(mathPrec), mathType, Int32(inputSize), Int32(hiddenSize), Int32(projSize), Int32(numLayers), cudnnDropoutDescriptor(dropout), UInt32(auxFlags)),
    # rnnForward parameters
    fwdMode::cudnnForwardMode_t = recording() ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE,
    devSeqLengths::DevArray{Cint,1},


)

end




cudnnRNNMathType(::Type{Float16})=CUDNN_TENSOR_OP_MATH
cudnnRNNMathType(::Type{Float32})=CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
cudnnRNNMathType(::Type{Float64})=CUDNN_DEFAULT_MATH
