module CUDNN

include("common.jl")
include("activation.jl")

## Forward functions:
# cudnnActivationForward
# cudnnBatchNormalizationForwardInference
# cudnnBatchNormalizationForwardTraining
# cudnnBatchNormalizationForwardTrainingEx
# cudnnConvolutionBiasActivationForward
# cudnnConvolutionForward
# cudnnDivisiveNormalizationForward
# cudnnDropoutForward
# cudnnLRNCrossChannelForward
# cudnnMultiHeadAttnForward
# cudnnNormalizationForwardInference  ## new in v8
# cudnnNormalizationForwardTraining   ## new in v8
# cudnnPoolingForward
# cudnnRNNForward  ## new in v8
# cudnnRNNForwardInference
# cudnnRNNForwardInferenceEx
# cudnnRNNForwardTraining
# cudnnRNNForwardTrainingEx
# cudnnSoftmaxForward
# cudnnSpatialTfGridGeneratorForward
# cudnnSpatialTfSamplerForward
# cudnnTransformTensor
# cudnnTransformTensorEx
# cudnnAddTensor
# foo-8.0.2/cudnn_ops_infer.h:cudnnOpTensor(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnReduceTensor(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnSetTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *valuePtr);
# foo-8.0.2/cudnn_ops_infer.h:cudnnScaleTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *alpha);
# foo-8.0.2/cudnn_ops_infer.h:cudnnTransformFilter(cudnnHandle_t handle,

end

## grep cudnnHandle_t

# foo-8.0.2/cudnn_adv_infer.h:cudnnSetRNNDescriptor_v6(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnGetRNNDescriptor_v6(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnRNNSetClip(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnRNNGetClip(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnSetRNNProjectionLayers(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnGetRNNProjectionLayers(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnBuildRNNDynamic(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int miniBatch);
# foo-8.0.2/cudnn_adv_infer.h:cudnnGetRNNWorkspaceSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnGetRNNTempSpaceSizes(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnGetRNNParamsSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnGetRNNWeightSpaceSize(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, size_t *weightSpaceSize);
# foo-8.0.2/cudnn_adv_infer.h:cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnGetRNNLinLayerBiasParams(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnGetRNNWeightParams(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnRNNForwardInference(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnRNNForwardInferenceEx(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnRNNForward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnAlgorithmDescriptor_t algoDesc);
# foo-8.0.2/cudnn_adv_infer.h:cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);
# foo-8.0.2/cudnn_adv_infer.h:cudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnGetMultiHeadAttnWeights(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_infer.h:cudnnMultiHeadAttnForward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:cudnnRNNForwardTraining(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:cudnnRNNBackwardData(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:cudnnRNNBackwardData_v8(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:cudnnRNNBackwardWeights(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:cudnnRNNBackwardWeights_v8(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:cudnnRNNForwardTrainingEx(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:cudnnRNNBackwardDataEx(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:cudnnRNNBackwardWeightsEx(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:cudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);
# foo-8.0.2/cudnn_adv_train.h:cudnnFindRNNForwardTrainingAlgorithmEx(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:cudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);
# foo-8.0.2/cudnn_adv_train.h:cudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);
# foo-8.0.2/cudnn_adv_train.h:cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:cudnnMultiHeadAttnBackwardData(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:cudnnMultiHeadAttnBackwardWeights(cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:    cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:    cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:    cudnnHandle_t handle,
# foo-8.0.2/cudnn_adv_train.h:    cudnnHandle_t handle,
# foo-8.0.2/cudnn_backend.h:cudnnBackendExecute(cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan, cudnnBackendDescriptor_t variantPack);
# foo-8.0.2/cudnn_cnn_infer.h:cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count);
# foo-8.0.2/cudnn_cnn_infer.h:cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_infer.h:cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_infer.h:cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_infer.h:cudnnIm2Col(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_infer.h:cudnnReorderFilterAndBias(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_infer.h:cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_infer.h:cudnnConvolutionForward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_infer.h:cudnnConvolutionBiasActivationForward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_infer.h:cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int *count);
# foo-8.0.2/cudnn_cnn_infer.h:cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_infer.h:cudnnFindConvolutionBackwardDataAlgorithmEx(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_infer.h:cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_infer.h:cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_infer.h:cudnnConvolutionBackwardData(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_infer.h:cudnnGetFoldedConvBackwardDataDescriptors(const cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_train.h:cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int *count);
# foo-8.0.2/cudnn_cnn_train.h:cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_train.h:cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_train.h:cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_train.h:cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_train.h:cudnnConvolutionBackwardFilter(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_train.h:cudnnConvolutionBackwardBias(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_train.h:cudnnMakeFusedOpsPlan(cudnnHandle_t handle,
# foo-8.0.2/cudnn_cnn_train.h:cudnnFusedOpsExecute(cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan, cudnnFusedOpsVariantParamPack_t varPack);
# foo-8.0.2/cudnn_ops_infer.h:cudnnQueryRuntimeError(cudnnHandle_t handle, cudnnStatus_t *rstatus, cudnnErrQueryMode_t mode, cudnnRuntimeTag_t *tag);
# foo-8.0.2/cudnn_ops_infer.h:cudnnDestroy(cudnnHandle_t handle);
# foo-8.0.2/cudnn_ops_infer.h:cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
# foo-8.0.2/cudnn_ops_infer.h:cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId);
# foo-8.0.2/cudnn_ops_infer.h:cudnnTransformTensor(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnTransformTensorEx(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnAddTensor(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnOpTensor(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnGetReductionIndicesSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnGetReductionWorkspaceSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnReduceTensor(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnSetTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *valuePtr);
# foo-8.0.2/cudnn_ops_infer.h:cudnnScaleTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *alpha);
# foo-8.0.2/cudnn_ops_infer.h:cudnnTransformFilter(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnSoftmaxForward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnPoolingForward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnActivationForward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnLRNCrossChannelForward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnDivisiveNormalizationForward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnBatchNormalizationForwardInference(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnNormalizationForwardInference(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnSpatialTfGridGeneratorForward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnSpatialTfSamplerForward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t *sizeInBytes);
# foo-8.0.2/cudnn_ops_infer.h:                          cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:                              cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:                          cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnDropoutForward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnGetAlgorithmSpaceSize(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, size_t *algoSpaceSizeInBytes);
# foo-8.0.2/cudnn_ops_infer.h:cudnnSaveAlgorithm(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:cudnnRestoreAlgorithm(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_infer.h:    cudnnHandle_t handle;   /* cudnn handle */
# foo-8.0.2/cudnn_ops_train.h:cudnnSoftmaxBackward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnPoolingBackward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnActivationBackward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnLRNCrossChannelBackward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnDivisiveNormalizationBackward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnGetBatchNormalizationBackwardExWorkspaceSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnGetBatchNormalizationTrainingExReserveSpaceSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:    cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:    cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnBatchNormalizationBackward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnBatchNormalizationBackwardEx(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnGetNormalizationForwardTrainingWorkspaceSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnGetNormalizationBackwardWorkspaceSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnGetNormalizationTrainingReserveSpaceSize(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnNormalizationForwardTraining(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnNormalizationBackward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnSpatialTfSamplerBackward(cudnnHandle_t handle,
# foo-8.0.2/cudnn_ops_train.h:cudnnDropoutBackward(cudnnHandle_t handle,
# (base) [dyuret@login03 cudnn]$ 
