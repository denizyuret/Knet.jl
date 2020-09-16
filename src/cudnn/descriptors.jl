using Base: @__doc__
using CUDA.CUDNN: 
    cudnnActivationDescriptor_t,
        cudnnCreateActivationDescriptor,
        cudnnSetActivationDescriptor,
        cudnnDestroyActivationDescriptor,
        cudnnActivationMode_t,
            CUDNN_ACTIVATION_SIGMOID,      # 0
            CUDNN_ACTIVATION_RELU,         # 1
            CUDNN_ACTIVATION_TANH,         # 2
            CUDNN_ACTIVATION_CLIPPED_RELU, # 3
            CUDNN_ACTIVATION_ELU,          # 4
            CUDNN_ACTIVATION_IDENTITY,     # 5
        cudnnNanPropagation_t,
            CUDNN_NOT_PROPAGATE_NAN, # 0
            CUDNN_PROPAGATE_NAN,     # 1
    cudnnAttnDescriptor_t,
        cudnnCreateAttnDescriptor,
        cudnnSetAttnDescriptor,
        cudnnDestroyAttnDescriptor,
        cudnnDataType_t,
            CUDNN_DATA_FLOAT,   # 0
            CUDNN_DATA_DOUBLE,  # 1
            CUDNN_DATA_HALF,    # 2
            CUDNN_DATA_INT8,    # 3
            CUDNN_DATA_INT32,   # 4
            CUDNN_DATA_INT8x4,  # 5
            CUDNN_DATA_UINT8,   # 6
            CUDNN_DATA_UINT8x4, # 7
            CUDNN_DATA_INT8x32, # 8
        cudnnMathType_t,
            CUDNN_DEFAULT_MATH,                    # 0
            CUDNN_TENSOR_OP_MATH,                  # 1
            CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION, # 2
            CUDNN_FMA_MATH,                        # 3
    cudnnCTCLossDescriptor_t,
        cudnnCreateCTCLossDescriptor,
        cudnnSetCTCLossDescriptor_v8,
        cudnnDestroyCTCLossDescriptor,
        cudnnLossNormalizationMode_t,
            CUDNN_LOSS_NORMALIZATION_NONE,    # 0
            CUDNN_LOSS_NORMALIZATION_SOFTMAX, # 1
    cudnnConvolutionDescriptor_t,
        cudnnCreateConvolutionDescriptor,
        cudnnSetConvolutionNdDescriptor,
        cudnnDestroyConvolutionDescriptor,
        cudnnConvolutionMode_t,
            CUDNN_CONVOLUTION,       # 0
            CUDNN_CROSS_CORRELATION, # 1
    cudnnDropoutDescriptor_t,
        cudnnCreateDropoutDescriptor,
        cudnnSetDropoutDescriptor,
        cudnnDestroyDropoutDescriptor,
    cudnnFilterDescriptor_t,
        cudnnCreateFilterDescriptor,
        cudnnSetFilterNdDescriptor,
        cudnnDestroyFilterDescriptor,
        cudnnTensorFormat_t,
            CUDNN_TENSOR_NCHW,        # 0, /* row major (wStride = 1, hStride = w) */
            CUDNN_TENSOR_NHWC,        # 1, /* feature maps interleaved ( cStride = 1 )*/
            CUDNN_TENSOR_NCHW_VECT_C, # 2, /* each image point is vector of element of C, vector length in data type */
    cudnnLRNDescriptor_t,
        cudnnCreateLRNDescriptor,
        cudnnSetLRNDescriptor,
        cudnnDestroyLRNDescriptor,
    cudnnOpTensorDescriptor_t,
        cudnnCreateOpTensorDescriptor,
        cudnnSetOpTensorDescriptor,
        cudnnDestroyOpTensorDescriptor,
        cudnnOpTensorOp_t,
            CUDNN_OP_TENSOR_ADD,  # 0
            CUDNN_OP_TENSOR_MUL,  # 1
            CUDNN_OP_TENSOR_MIN,  # 2
            CUDNN_OP_TENSOR_MAX,  # 3
            CUDNN_OP_TENSOR_SQRT, # 4
            CUDNN_OP_TENSOR_NOT,  # 5
    cudnnPoolingDescriptor_t,
        cudnnCreatePoolingDescriptor,
        cudnnSetPoolingNdDescriptor,
        cudnnDestroyPoolingDescriptor,
        cudnnPoolingMode_t,
            CUDNN_POOLING_MAX,                           # 0,
            CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, # 1, /* count for average includes padded values */
            CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, # 2, /* count for average does not include padded values */
            CUDNN_POOLING_MAX_DETERMINISTIC,             # 3
    cudnnRNNDescriptor_t,
        cudnnCreateRNNDescriptor,
        cudnnSetRNNDescriptor,
        cudnnDestroyRNNDescriptor,
        cudnnRNNAlgo_t,
            CUDNN_RNN_ALGO_STANDARD,        # 0
            CUDNN_RNN_ALGO_PERSIST_STATIC,  # 1
            CUDNN_RNN_ALGO_PERSIST_DYNAMIC, # 2
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
    cudnnRNNDataDescriptor_t,
        cudnnCreateRNNDataDescriptor,
        cudnnSetRNNDataDescriptor,
        cudnnDestroyRNNDataDescriptor,
        cudnnRNNDataLayout_t,
            CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,   # 0, /* padded, outer stride from one time-step to the next */
            CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,     # 1, /* sequence length sorted and packed as in basic RNN api */
            CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED, # 2, /* padded, outer stride from one batch to the next */
    cudnnReduceTensorDescriptor_t,
        cudnnCreateReduceTensorDescriptor,
        cudnnSetReduceTensorDescriptor,
        cudnnDestroyReduceTensorDescriptor,
        cudnnReduceTensorOp_t,
            CUDNN_REDUCE_TENSOR_ADD,          # 0
            CUDNN_REDUCE_TENSOR_MUL,          # 1
            CUDNN_REDUCE_TENSOR_MIN,          # 2
            CUDNN_REDUCE_TENSOR_MAX,          # 3
            CUDNN_REDUCE_TENSOR_AMAX,         # 4
            CUDNN_REDUCE_TENSOR_AVG,          # 5
            CUDNN_REDUCE_TENSOR_NORM1,        # 6
            CUDNN_REDUCE_TENSOR_NORM2,        # 7
            CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS, # 8
        cudnnReduceTensorIndices_t,
            CUDNN_REDUCE_TENSOR_NO_INDICES,        # 0
            CUDNN_REDUCE_TENSOR_FLATTENED_INDICES, # 1
        cudnnIndicesType_t,
            CUDNN_32BIT_INDICES, # 0
            CUDNN_64BIT_INDICES, # 1
            CUDNN_16BIT_INDICES, # 2
            CUDNN_8BIT_INDICES,  # 3
    cudnnSeqDataDescriptor_t,
        cudnnCreateSeqDataDescriptor,
        cudnnSetSeqDataDescriptor,
        cudnnDestroySeqDataDescriptor,
        cudnnSeqDataAxis_t,
            CUDNN_SEQDATA_TIME_DIM,  # 0, /* index in time */
            CUDNN_SEQDATA_BATCH_DIM, # 1, /* index in batch */
            CUDNN_SEQDATA_BEAM_DIM,  # 2, /* index in beam */
            CUDNN_SEQDATA_VECT_DIM,  # 3  /* index in vector */
    cudnnSpatialTransformerDescriptor_t,
        cudnnCreateSpatialTransformerDescriptor,
        cudnnSetSpatialTransformerNdDescriptor,
        cudnnDestroySpatialTransformerDescriptor,
        cudnnSamplerType_t,
            CUDNN_SAMPLER_BILINEAR, # 0
    cudnnTensorDescriptor_t,
        cudnnCreateTensorDescriptor,
        cudnnSetTensorNdDescriptorEx,
        cudnnDestroyTensorDescriptor,
    cudnnTensorTransformDescriptor_t,
        cudnnCreateTensorTransformDescriptor,
        cudnnSetTensorTransformDescriptor,
        cudnnDestroyTensorTransformDescriptor,
        cudnnFoldingDirection_t,
            CUDNN_TRANSFORM_FOLD,   # 0U,
            CUDNN_TRANSFORM_UNFOLD, # 1U,
    handle


"""
    @cudnnDescriptor(XXX, setter=cudnnSetXXXDescriptor)

Defines a new type `cudnnXXXDescriptor` with a single field `ptr::cudnnXXXDescriptor_t` and
its constructor. The second optional argument is the function that sets the descriptor
fields and defaults to `cudnnSetXXXDescriptor`. The constructor is memoized, i.e. when
called with the same arguments it returns the same object rather than creating a new one.

The arguments of the constructor and thus the keys to the memoization cache depend on the
setter: If the setter has arguments `cudnnSetXXXDescriptor(ptr::cudnnXXXDescriptor_t,
args...)`, then the constructor has `cudnnXXXDescriptor(args...)`. The user can control
these arguments by defining a custom setter.
"""
macro cudnnDescriptor(x, set = Symbol("cudnnSet$(x)Descriptor"))
    sname = Symbol("cudnn$(x)Descriptor")
    tname = Symbol("cudnn$(x)Descriptor_t")
    cache = Symbol("cudnn$(x)DescriptorCache")
    create = Symbol("cudnnCreate$(x)Descriptor")
    destroy = Symbol("cudnnDestroy$(x)Descriptor")
    return quote
        @__doc__ mutable struct $sname; ptr::$tname; end    # needs to be mutable for finalizer
        Base.unsafe_convert(::Type{<:Ptr}, d::$sname)=d.ptr # needed for ccalls
        const $cache = Dict{Tuple,$sname}()                 # Dict is 3x faster than IdDict!
        function $sname(args...)
            get!($cache, args) do
                ptr = $tname[C_NULL]
                $create(ptr)
                $set(ptr[1], args...)
                d = $sname(ptr[1])
                finalizer(x->$destroy(x.ptr), d)
                return d
            end
        end
    end |> esc
end


"""
    cudnnActivationDescriptor(mode::cudnnActivationMode_t, 
                              reluNanOpt::cudnnNanPropagation_t,
                              coef::Cfloat)
"""
@cudnnDescriptor(Activation)


"""
    cudnnAttnDescriptor(attnMode::Cuint,
                        nHeads::Cint,
                        smScaler::Cdouble,
                        dataType::cudnnDataType_t,
                        computePrec::cudnnDataType_t,
                        mathType::cudnnMathType_t,
                        attnDropoutDesc::cudnnDropoutDescriptor_t,
                        postDropoutDesc::cudnnDropoutDescriptor_t,
                        qSize::Cint,
                        kSize::Cint,
                        vSize::Cint,
                        qProjSize::Cint,
                        kProjSize::Cint,
                        vProjSize::Cint,
                        oProjSize::Cint,
                        qoMaxSeqLength::Cint,
                        kvMaxSeqLength::Cint,
                        maxBatchSize::Cint,
                        maxBeamSize::Cint)
"""
@cudnnDescriptor(Attn)


"""
    cudnnCTCLossDescriptor(compType::cudnnDataType_t,
                           normMode::cudnnLossNormalizationMode_t,
                           gradMode::cudnnNanPropagation_t,
                           maxLabelLength::Cint)
"""
@cudnnDescriptor(CTCLoss, cudnnSetCTCLossDescriptor_v8)


"""
    cudnnConvolutionDescriptor(arrayLength::Cint,
                               padA::Vector{Cint},
                               filterStrideA::Vector{Cint},
                               dilationA::Vector{Cint},
                               mode::cudnnConvolutionMode_t,
                               computeType::cudnnDataType_t)
"""
@cudnnDescriptor(Convolution, cudnnSetConvolutionNdDescriptor)


"""
    cudnnDropoutDescriptor(dropout::Cfloat)
"""
@cudnnDescriptor(Dropout, cudnnSetDropoutDescriptorFromFloat)


"""
    cudnnFilterDescriptor(dataType::cudnnDataType_t,
                          format::cudnnTensorFormat_t,
                          nbDims::Cint,
                          filterDimA::Vector{Cint})
"""
@cudnnDescriptor(Filter, cudnnSetFilterNdDescriptor)


"""
    cudnnLRNDescriptor(lrnN::Cuint,
                       lrnAlpha::Cdouble, 
                       lrnBeta::Cdouble, 
                       lrnK::Cdouble)
"""
@cudnnDescriptor(LRN)


"""
    cudnnOpTensorDescriptor(opTensorOp::cudnnOpTensorOp_t,
                            opTensorCompType::cudnnDataType_t,
                            opTensorNanOpt::cudnnNanPropagation_t)
"""
@cudnnDescriptor(OpTensor)


"""
    cudnnPoolingDescriptor(mode::cudnnPoolingMode_t,
                           maxpoolingNanOpt::cudnnNanPropagation_t,
                           nbDims::Cint,
                           windowDimA::Vector{Cint},
                           paddingA::Vector{Cint},
                           strideA::Vector{Cint})
"""
@cudnnDescriptor(Pooling, cudnnSetPoolingNdDescriptor)


"""
    cudnnRNNDescriptor(algo::cudnnRNNAlgo_t,
                       cellMode::cudnnRNNMode_t,
                       biasMode::cudnnRNNBiasMode_t,
                       dirMode::cudnnDirectionMode_t,
                       inputMode::cudnnRNNInputMode_t,
                       dataType::cudnnDataType_t,
                       mathPrec::cudnnDataType_t,
                       mathType::cudnnMathType_t,
                       inputSize::Int32,
                       hiddenSize::Int32,
                       projSize::Int32,
                       numLayers::Int32,
                       dropoutDesc::cudnnDropoutDescriptor_t,
                       auxFlags::UInt32)
"""
@cudnnDescriptor(RNN, cudnnSetRNNDescriptor_v8)


"""
    cudnnRNNDataDescriptor(dataType::cudnnDataType_t,
                           layout::cudnnRNNDataLayout_t,
                           maxSeqLength::Cint,
                           batchSize::Cint,
                           vectorSize::Cint,
                           seqLengthArray::Vector{Cint},
                           paddingFill::Ptr{Cvoid})
"""
@cudnnDescriptor(RNNData)


"""
    cudnnReduceTensorDescriptor(reduceTensorOp::cudnnReduceTensorOp_t,
                                reduceTensorCompType::cudnnDataType_t,
                                reduceTensorNanOpt::cudnnNanPropagation_t,
                                reduceTensorIndices::cudnnReduceTensorIndices_t,
                                reduceTensorIndicesType::cudnnIndicesType_t)
"""
@cudnnDescriptor(ReduceTensor)


"""
    cudnnSeqDataDescriptor(dataType::cudnnDataType_t,
                           nbDims::Cint,
                           dimA::Vector{Cint},
                           axes::Vector{cudnnSeqDataAxis_t},
                           seqLengthArraySize::Csize_t,
                           seqLengthArray::Vector{Cint},
                           paddingFill::Ptr{Cvoid})
"""
@cudnnDescriptor(SeqData)


"""
    cudnnSpatialTransformerDescriptor(samplerType::cudnnSamplerType_t,
                                      dataType::cudnnDataType_t,
                                      nbDims::Cint,
                                      dimA::Vector{Cint})
"""
@cudnnDescriptor(SpatialTransformer, cudnnSetSpatialTransformerNdDescriptor)


"""
    cudnnTensorDescriptor(format::cudnnTensorFormat_t,
                          dataType::cudnnDataType_t,
                          nbDims::Cint,
                          dimA::Vector{Cint})
"""
@cudnnDescriptor(Tensor, cudnnSetTensorNdDescriptorEx)


"""
    cudnnTensorTransformDescriptor(nbDims::UInt32,
                                   destFormat::cudnnTensorFormat_t,
                                   padBeforeA::Vector{Int32},
                                   padAfterA::Vector{Int32},
                                   foldA::Vector{UInt32},
                                   direction::cudnnFoldingDirection_t)
"""
@cudnnDescriptor(TensorTransform)
