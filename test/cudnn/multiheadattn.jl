using Test, Random, CUDA, Knet, AutoGrad

using CUDA.CUDNN: 
    cudnnMultiHeadAttnForward,
    cudnnMultiHeadAttnForward!,
    cudnnMultiHeadAttnBackwardData,
    cudnnMultiHeadAttnBackwardWeights,
    cudnnGetMultiHeadAttnBuffers,
    cudnnGetMultiHeadAttnWeights,
    cudnnAttnDescriptor,
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
    cudnnSeqDataDescriptor,
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
    cudnnDataType,
    cudnnSeqDataDefaultAxes,
    math_mode,
    sdim,
    handle


@testset "cudnn/multiheadattn" begin

    Q,K,V,B,T,F = 6,6,5,4,3,Float64
    w, q, k, v = (Param(CUDA.randn(x...)) for x in ((F,400),(F,Q,B,T),(F,K,B,T),(F,V,B,T)))

    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; )


    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; seqLengthsQO = Cint[1,2,3,1])
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; seqLengthsKV = Cint[1,2,3,1])
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; nHeads = 2)
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; smScaler = 2)
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; mathType = CUDNN_DEFAULT_MATH)
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; mathType = CUDNN_TENSOR_OP_MATH)
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; mathType = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; mathType = CUDNN_FMA_MATH)
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; kProjSize = 7, qProjSize = 7) # k and q have to match
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; vProjSize = 7)
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; oProjSize = 7)
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; qoMaxSeqLength = 7)
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; kvMaxSeqLength = 7)
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; maxBatchSize = 7)
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; maxBeamSize = 7)
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; loWinIdx = fill(Cint(1),T))
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; hiWinIdx = fill(Cint(1),T))
    # currIdx >= 0 can only be used in inference mode
    # @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; currIdx = 0)

    # Test residuals: residuals and output (and thus v unless oProjSize>0) must match q in vector size
    v, residuals = (Param(CUDA.randn(x...)) for x in ((F,Q,B,T),(F,Q,B,T)))
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; residuals = residuals)

    # PROJ_BIASES not supposed to be supported according to docs, it works for some cases
    # (attnDesc.attnMode & CUDNN_ATTN_ENABLE_PROJ_BIASES == 0) || (@warn "The CUDNN_ATTN_ENABLE_PROJ_BIASES option is not supported in the multi-head attention gradient functions." maxlog=1)
    @test_skip @gcheck cudnnMultiHeadAttnForward(w, q, k, v; attnMode = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | CUDNN_ATTN_ENABLE_PROJ_BIASES |> Cuint, qProjSize=9, kProjSize=9, vProjSize=6, oProjSize=4)

    # Test nonstandard axes order
    w, q, k, v = (Param(CUDA.randn(x...)) for x in ((F,100),(F,Q,T,B),(F,K,T,B),(F,V,T,B)))
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; axes = [CUDNN_SEQDATA_VECT_DIM, CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_BEAM_DIM])

    # Test beam handling
    w, q, k, v = (Param(CUDA.randn(x...)) for x in ((F,100),(F,Q,B,T,2),(F,K,B,T,1),(F,V,B,T,1)))
    @test @gcheck cudnnMultiHeadAttnForward(w, q, k, v; )
    # CUDNN_ATTN_QUERYMAP_ONE_TO_ONE does not seem to be supported
    w, q, k, v = (Param(CUDA.randn(x...)) for x in ((F,100),(F,Q,B,T,2),(F,K,B,T,2),(F,V,B,T,2)))
    @test_skip @gcheck cudnnMultiHeadAttnForward(w, q, k, v; attnMode = CUDNN_ATTN_QUERYMAP_ONE_TO_ONE | CUDNN_ATTN_DISABLE_PROJ_BIASES |> Cuint) ## Not supported

end
