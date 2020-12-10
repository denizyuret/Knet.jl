using Test, CUDA, Random, Knet, AutoGrad
using CUDA.CUDNN:
    cudnnConvolutionForward,
    cudnnConvolutionForward!,
    cudnnConvolutionBackwardFilter,
    cudnnConvolutionBackwardData,
    cudnnGetConvolutionNdForwardOutputDim,
    cudnnSetConvolutionMathType,
    cudnnSetConvolutionReorderType,
    cudnnSetConvolutionGroupCount,
    cudnnFindConvolutionForwardAlgorithmEx,
        cudnnConvolutionFwdAlgoPerf_t,
    cudnnFindConvolutionBackwardFilterAlgorithmEx,
        cudnnConvolutionBwdFilterAlgoPerf_t,
    cudnnFindConvolutionBackwardDataAlgorithmEx,
        cudnnConvolutionBwdDataAlgoPerf_t,
    cudnnConvolutionDescriptor,
        cudnnConvolutionDescriptor_t,
        cudnnCreateConvolutionDescriptor,
        cudnnSetConvolutionNdDescriptor,
        cudnnDestroyConvolutionDescriptor,
    cudnnConvolutionMode_t,
        CUDNN_CONVOLUTION,       # 0
        CUDNN_CROSS_CORRELATION, # 1
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
    cudnnMathType_t,
        CUDNN_DEFAULT_MATH,                    # 0
        CUDNN_TENSOR_OP_MATH,                  # 1
        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION, # 2
        CUDNN_FMA_MATH,                        # 3
    cudnnReorderType_t,
        CUDNN_DEFAULT_REORDER, # 0
        CUDNN_NO_REORDER,      # 1
    cudnnConvolutionFwdAlgo_t,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,         # 0
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, # 1
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,                  # 2
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,                # 3
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,                   # 4
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,            # 5
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,              # 6
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,     # 7
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT,                 # 8
    cudnnConvolutionBwdFilterAlgo_t,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,                 # 0, /* non-deterministic */
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,                 # 1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,               # 2,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,                 # 3, /* non-deterministic */
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD,          # 4, /* not implemented */
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED, # 5,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,        # 6,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,             # 7
    cudnnConvolutionBwdDataAlgo_t,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,                 # 0, /* non-deterministic */
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,                 # 1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,               # 2,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,        # 3,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,          # 4,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED, # 5,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT,             # 6
    cudnnDataType,
    convdims,
    math_mode,
    handle


@testset "cudnn/convolution" begin
    T = Float64
    ax,aw,ag,ab = randn(T,8,8,4,4),randn(T,3,3,4,4),randn(T,3,3,2,4),randn(T,1,1,4,1)
    cx,cw,cg,cb = Param.(CuArray.((ax,aw,ag,ab)))
    cz = Param(cudnnConvolutionForward(cw,cx))

    # These call cudnnConvolutionForward
    @test @gcheck cudnnConvolutionForward(cw,cx)
    @test @gcheck cudnnConvolutionForward(cw,cx;padding=1)
    @test @gcheck cudnnConvolutionForward(cw,cx;stride=2)
    @test @gcheck cudnnConvolutionForward(cw,cx;dilation=2)
    @test @gcheck cudnnConvolutionForward(cg,cx;group=2)
    @test @gcheck cudnnConvolutionForward(cw,cx;mathType=CUDNN_DEFAULT_MATH)
    @test @gcheck cudnnConvolutionForward(cw,cx;mathType=CUDNN_TENSOR_OP_MATH)
    @test @gcheck cudnnConvolutionForward(cw,cx;mathType=CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
    @test @gcheck cudnnConvolutionForward(cw,cx;reorderType=CUDNN_NO_REORDER)
    @test @gcheck cudnnConvolutionForward(cw,cx;alpha=2)

    # These call cudnnConvolutionBiasActivationForward
    @test @gcheck cudnnConvolutionForward(cw,cx;bias=cb)
    @test @gcheck cudnnConvolutionForward(cw,cx;z=cz)
    @test @gcheck cudnnConvolutionForward(cw,cx;z=cz,beta=2)
    @test @gcheck cudnnConvolutionForward(cw,cx;activation=CUDNN_ACTIVATION_RELU)
    @test @gcheck cudnnConvolutionForward(cw,cx;bias=cb,z=cz)
    @test @gcheck cudnnConvolutionForward(cw,cx;bias=cb,activation=CUDNN_ACTIVATION_RELU)
    @test @gcheck cudnnConvolutionForward(cw,cx;bias=cb,padding=1)
    @test @gcheck cudnnConvolutionForward(cw,cx;bias=cb,stride=2)
    @test @gcheck cudnnConvolutionForward(cw,cx;bias=cb,dilation=2)
    @test @gcheck cudnnConvolutionForward(cg,cx;bias=cb,group=2)
    @test @gcheck cudnnConvolutionForward(cw,cx;bias=cb,mathType=CUDNN_DEFAULT_MATH)
    @test @gcheck cudnnConvolutionForward(cw,cx;bias=cb,mathType=CUDNN_TENSOR_OP_MATH)
    @test @gcheck cudnnConvolutionForward(cw,cx;bias=cb,mathType=CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
    @test @gcheck cudnnConvolutionForward(cw,cx;bias=cb,reorderType=CUDNN_NO_REORDER)
    @test @gcheck cudnnConvolutionForward(cw,cx;bias=cb,alpha=2)
    @test @gcheck cudnnConvolutionForward(cw,cx;bias=cb,beta=2,z=cz)
end
