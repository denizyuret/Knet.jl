let convDesc=nothing
    global conv4
    function conv4(w::KnetArray, x::KnetArray;
                   padding=0, stride=1, upscale=1, mode=0, algo=0,
                   workSpace=nothing, workSpaceSizeInBytes=0,
                   alpha=1.0, beta=0.0)
        cudnnCheck(ccall((:cudnnConvolutionForward,"libcudnn"),cudnnStatus_t,
                         (cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionFwdAlgo_t,Ptr{Void},Csize_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),
                         cudnnhandle,alpha,xDesc,x,wDesc,w,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,yDesc,y))
    end
end
