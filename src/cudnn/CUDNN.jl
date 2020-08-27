module CUDNN

include("cudnn_retry.jl")       # TODO: export this to Ops20_gpu, auto-add unsafe_cudnn?
include("common.jl")
include("activation.jl")
include("dropout.jl")
include("multiheadattn.jl")

end

## grep cudnnHandle_t
# add8: cudnnBackendExecute
# add8: cudnnBuildRNNDynamic
# add8: cudnnCTCLoss_v8
# add8: cudnnGetFoldedConvBackwardDataDescriptors
# add8: cudnnGetNormalizationBackwardWorkspaceSize
# add8: cudnnGetNormalizationForwardTrainingWorkspaceSize
# add8: cudnnGetNormalizationTrainingReserveSpaceSize
# add8: cudnnGetRNNTempSpaceSizes
# add8: cudnnGetRNNWeightParams
# add8: cudnnGetRNNWeightSpaceSize
# add8: cudnnNormalizationBackward
# add8: cudnnNormalizationForwardInference
# add8: cudnnNormalizationForwardTraining
# add8: cudnnRNNBackwardData_v8
# add8: cudnnRNNBackwardWeights_v8
# add8: cudnnRNNForward
# cudnnActivationBackward
# cudnnActivationForward
# cudnnAddTensor
# cudnnBatchNormalizationBackward
# cudnnBatchNormalizationBackwardEx
# cudnnBatchNormalizationForwardInference
# cudnnBatchNormalizationForwardTraining
# cudnnBatchNormalizationForwardTrainingEx
# cudnnConvolutionBackwardBias
# cudnnConvolutionBackwardData
# cudnnConvolutionBackwardFilter
# cudnnConvolutionBiasActivationForward
# cudnnConvolutionForward
# cudnnCreate
# cudnnCTCLoss
# cudnnDestroy
# cudnnDivisiveNormalizationBackward
# cudnnDivisiveNormalizationForward
# cudnnDropoutBackward
# cudnnDropoutForward
# cudnnDropoutGetStatesSize
# cudnnFindConvolutionBackwardDataAlgorithm
# cudnnFindConvolutionBackwardDataAlgorithmEx
# cudnnFindConvolutionBackwardFilterAlgorithm
# cudnnFindConvolutionBackwardFilterAlgorithmEx
# cudnnFindConvolutionForwardAlgorithm
# cudnnFindConvolutionForwardAlgorithmEx
# cudnnFusedOpsExecute
# cudnnGetBatchNormalizationBackwardExWorkspaceSize
# cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize
# cudnnGetBatchNormalizationTrainingExReserveSpaceSize
# cudnnGetConvolutionBackwardDataAlgorithm_v7
# cudnnGetConvolutionBackwardDataAlgorithmMaxCount
# cudnnGetConvolutionBackwardDataWorkspaceSize
# cudnnGetConvolutionBackwardFilterAlgorithm_v7
# cudnnGetConvolutionBackwardFilterAlgorithmMaxCount
# cudnnGetConvolutionBackwardFilterWorkspaceSize
# cudnnGetConvolutionForwardAlgorithm_v7
# cudnnGetConvolutionForwardAlgorithmMaxCount
# cudnnGetConvolutionForwardWorkspaceSize
# cudnnGetCTCLossWorkspaceSize
# cudnnGetCTCLossWorkspaceSize_v8
# cudnnGetDropoutDescriptor
# cudnnGetMultiHeadAttnBuffers
# cudnnGetMultiHeadAttnWeights
# cudnnGetReductionIndicesSize
# cudnnGetReductionWorkspaceSize
# cudnnGetStream
# cudnnIm2Col
# cudnnLRNCrossChannelBackward
# cudnnLRNCrossChannelForward
# cudnnMakeFusedOpsPlan
# cudnnMultiHeadAttnBackwardData
# cudnnMultiHeadAttnBackwardWeights
# cudnnMultiHeadAttnForward
# cudnnOpTensor
# cudnnPoolingBackward
# cudnnPoolingForward
# cudnnQueryRuntimeError
# cudnnReduceTensor
# cudnnReorderFilterAndBias
# cudnnRestoreDropoutDescriptor
# cudnnScaleTensor
# cudnnSetDropoutDescriptor
# cudnnSetStream
# cudnnSetTensor
# cudnnSoftmaxBackward
# cudnnSoftmaxForward
# cudnnSpatialTfGridGeneratorBackward
# cudnnSpatialTfGridGeneratorForward
# cudnnSpatialTfSamplerBackward
# cudnnSpatialTfSamplerForward
# cudnnTransformFilter
# cudnnTransformTensor
# cudnnTransformTensorEx
# dep8: cudnnFindRNNBackwardDataAlgorithmEx
# dep8: cudnnFindRNNBackwardWeightsAlgorithmEx
# dep8: cudnnFindRNNForwardInferenceAlgorithmEx
# dep8: cudnnFindRNNForwardTrainingAlgorithmEx
# dep8: cudnnGetAlgorithmSpaceSize
# dep8: cudnnGetRNNBackwardDataAlgorithmMaxCount
# dep8: cudnnGetRNNBackwardWeightsAlgorithmMaxCount
# dep8: cudnnGetRNNDescriptor_v6
# dep8: cudnnGetRNNForwardInferenceAlgorithmMaxCount
# dep8: cudnnGetRNNForwardTrainingAlgorithmMaxCount
# dep8: cudnnGetRNNLinLayerBiasParams
# dep8: cudnnGetRNNLinLayerMatrixParams
# dep8: cudnnGetRNNParamsSize
# dep8: cudnnGetRNNProjectionLayers
# dep8: cudnnGetRNNTrainingReserveSize
# dep8: cudnnGetRNNWorkspaceSize
# dep8: cudnnRestoreAlgorithm
# dep8: cudnnRNNBackwardData
# dep8: cudnnRNNBackwardDataEx
# dep8: cudnnRNNBackwardWeights
# dep8: cudnnRNNBackwardWeightsEx
# dep8: cudnnRNNForwardInference
# dep8: cudnnRNNForwardInferenceEx
# dep8: cudnnRNNForwardTraining
# dep8: cudnnRNNForwardTrainingEx
# dep8: cudnnRNNGetClip
# dep8: cudnnRNNSetClip
# dep8: cudnnSaveAlgorithm
# dep8: cudnnSetRNNAlgorithmDescriptor
# dep8: cudnnSetRNNDescriptor_v6
# dep8: cudnnSetRNNProjectionLayers
