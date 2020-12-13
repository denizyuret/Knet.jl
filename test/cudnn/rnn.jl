using Test, CUDA, Random, Knet, AutoGrad
using CUDA.CUDNN:
    cudnnRNNForward,
    cudnnRNNForward!,
    cudnnRNNBackwardData_v8,
    cudnnRNNBackwardWeights_v8,
    cudnnRNNDescriptor,
    cudnnRNNDescriptor_t,
    cudnnSetRNNDescriptor_v8,
    cudnnGetRNNWeightSpaceSize,
    cudnnGetRNNTempSpaceSizes,
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
        CUDNN_FWD_MODE_INFERENCE, # 0
        CUDNN_FWD_MODE_TRAINING,  # 1
    cudnnRNNDataDescriptor_t,
    cudnnSetRNNDataDescriptor,
    cudnnRNNDataLayout_t,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,   # 0, /* padded, outer stride from one time-step to the next */
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,     # 1, /* sequence length sorted and packed as in basic RNN api */
        CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED, # 2, /* padded, outer stride from one batch to the next */
    cudnnWgradMode_t,
        CUDNN_WGRAD_MODE_ADD, # 0, /* add partial gradients to wgrad output buffers */
        CUDNN_WGRAD_MODE_SET, # 1, /* write partial gradients to wgrad output buffers */
    cudnnTensorDescriptor,
    cudnnDropoutDescriptor,
    cudnnDataType,
    math_mode,
    handle


@testset "cudnn/rnn" begin
    X,H,B,T,L,F = 4,4,4,4,1,Float64
    aw,ax,ah,ac = randn(F,10000), randn(F,X,B,T), randn(F,H,B,L), randn(F,H,B,L)
    cw,cx,ch,cc = Param.(CuArray.((aw,ax,ah,ac)))

    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,hx=ch)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,cx=cc)
    @test @gcheck (hy=Ref{Any}(); y=cudnnRNNForward(cw,cx;hiddenSize=H,hy); sum(y)+sum(hy[]))
    @test @gcheck (cy=Ref{Any}(); y=cudnnRNNForward(cw,cx;hiddenSize=H,cy); sum(y)+sum(cy[]))
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,layout=CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,layout=CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,seqLengthArray=Cint[1,2,1,2])
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,fwdMode=CUDNN_FWD_MODE_TRAINING)
    #@test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,algo=CUDNN_RNN_ALGO_PERSIST_STATIC)  # not supported
    #@test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,algo=CUDNN_RNN_ALGO_PERSIST_DYNAMIC) # causes segfault
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,cellMode=CUDNN_RNN_RELU)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,cellMode=CUDNN_RNN_TANH)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,cellMode=CUDNN_GRU)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,biasMode=CUDNN_RNN_NO_BIAS)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,biasMode=CUDNN_RNN_SINGLE_INP_BIAS)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,biasMode=CUDNN_RNN_SINGLE_REC_BIAS)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,dirMode=CUDNN_BIDIRECTIONAL)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,inputMode=CUDNN_SKIP_INPUT)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,mathPrec=Float64)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,mathType=CUDNN_DEFAULT_MATH)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,mathType=CUDNN_TENSOR_OP_MATH)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,mathType=CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,projSize=H-1)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,numLayers=2)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,dropout=0.5)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,auxFlags=CUDNN_RNN_PADDED_IO_DISABLED)
    @test @gcheck cudnnRNNForward(cw,cx;hiddenSize=H,auxFlags=CUDNN_RNN_PADDED_IO_ENABLED)
end
