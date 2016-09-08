typealias CPtr Ptr{Void}

function conv4{T}(w::KnetArray{T}, x::KnetArray{T};
                  padding=0, stride=1, upscale=1, mode=0, 
                  algo=0, workSpace=nothing, workSpaceSizeInBytes=0,
                  alpha=1.0, beta=0.0)
    convDesc = cudnnCreateConvolutionDescriptor(x, padding, stride, upscale, mode)
    xDesc = cudnnCreateTensorDescriptor(x)
    wDesc = cudnnCreateFilterDescriptor(w)
    yDims = Array(Cint,ndims(x))
    cudnnCheck(ccall((:cudnnGetConvolutionNdForwardOutputDim,"libcudnn"),UInt32,
                     (CPtr,CPtr,CPtr,Cint,Cptr), convDesc,xDesc,wDesc,ndims(x),yDims))
    y = KnetArray(T, reverse(yDims)...)
    yDesc = cudnnCreateTensorDescriptor(y)
    cudnnCheck(ccall((:cudnnConvolutionForward,"libcudnn"),UInt32,
                     (CPtr,CPtr,CPtr,CPtr,CPtr,CPtr,CPtr,UInt32,CPtr,Csize_t,CPtr,CPtr,CPtr),
                     cudnnhandle,T[alpha],xDesc,x,wDesc,w,convDesc,algo,workSpace,workSpaceSizeInBytes,T[beta],yDesc,y))
    cudnnDestroyTensorDescriptor(xDesc)
    cudnnDestroyTensorDescriptor(yDesc)
    cudnnDestroyFilterDescriptor(wDesc)
    cudnnDestroyConvolutionDescriptor(convDesc)
    return y
end

# do we need to do a special bias or is cuda12 enough?

@primitive conv4(w,x;o...),dy  conv4back(x,dy)

# conv4back
# 8 descriptor functions
# pool and poolback
