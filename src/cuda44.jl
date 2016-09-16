# This hack should go away when I implement direct cudnn support.
if Pkg.installed("CUDNN") != nothing

eval(Expr(:using,:CUDArt))
eval(Expr(:using,:CUDNN))

# Define some new primitives: conv4 and pool

# CUDNN supports CudaArrays, here is a hack until we implement direct KnetArray support.
# It creates a CudaArray with space shared with a KnetArray.

CUDArt.CudaArray{T,N}(x::KnetArray{T,N})=CudaArray{T,N}(CudaPtr{T}(x.ptr.ptr), size(x), x.ptr.dev)

# The names conv and conv2 already taken in Base.  I will use conv4 for the cudnn version.

function conv4{T}(w::KnetArray{T},x::KnetArray{T}; o...)
    cx = CudaArray(x)
    cw = CudaArray(w)
    ydims = cudnnGetConvolutionNdForwardOutputDim(cx,cw; o...)
    y = similar(x, ydims)
    cy = CudaArray(y)
    cudnnConvolutionForward(cx, cw, cy; o...)
    return y
end

function conv4x{T}(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T}; o...)
    dx = similar(x)
    cw = CudaArray(w)
    cdx = CudaArray(dx)
    cdy = CudaArray(dy)
    cudnnConvolutionBackwardData(cw,cdy,cdx; o...)
    return dx
end

function conv4w{T}(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T}; o...)
    dw = similar(w)
    cx = CudaArray(x)
    cdy = CudaArray(dy)
    cdw = CudaArray(dw)
    cudnnConvolutionBackwardFilter(cx,cdy,cdw; o...)
    return dw
end

@primitive  conv4(w,x; o...),dy  conv4w(w,x,dy;o...)  conv4x(w,x,dy;o...)
@zerograd conv4x(w,x,dy;o...)
@zerograd conv4w(w,x,dy;o...)

function pool{T}(x::KnetArray{T}; o...)
    pd = CUDNN.PD(ndims=ndims(x), o...)
    cx = CudaArray(x)
    ydims = cudnnGetPoolingNdForwardOutputDim(pd, cx)
    y = similar(x, ydims)
    cy = CudaArray(y)
    cudnnPoolingForward(cx, cy; o...)
    return y
end

function poolx{T}(x::KnetArray{T},y::KnetArray{T},dy::KnetArray{T}; o...)
    dx = similar(x)
    cx = CudaArray(x)
    cy = CudaArray(y)
    cdy = CudaArray(dy)
    cdx = CudaArray(dx)
    cudnnPoolingBackward(cy,cdy,cx,cdx; o...)
    return dx
end

@primitive  pool(x;o...),dy,y  poolx(x,y,dy;o...)
@zerograd poolx(x,y,dy;o...)


else # if Pkg.installed("CUDNN") != nothing

conv4(x...; o...) = error("For GPU convolution support please use: Pkg.add(\"CUDNN\"); Pkg.build(\"Knet\")")
pool(x...; o...)  = error("For GPU convolution support please use: Pkg.add(\"CUDNN\"); Pkg.build(\"Knet\")")

end

# Work in progress on direct KnetArray support to eliminate dependency on CUDNN:

# typealias CPtr Ptr{Void}

# function conv4{T}(w::KnetArray{T}, x::KnetArray{T};
#                   padding=0, stride=1, upscale=1, mode=0, 
#                   algo=0, workSpace=nothing, workSpaceSizeInBytes=0,
#                   alpha=1.0, beta=0.0)
#     convDesc = cudnnCreateConvolutionDescriptor(x, padding, stride, upscale, mode)
#     xDesc = cudnnCreateTensorDescriptor(x)
#     wDesc = cudnnCreateFilterDescriptor(w)
#     yDims = Array(Cint,ndims(x))
#     cudnnCheck(ccall((:cudnnGetConvolutionNdForwardOutputDim,"libcudnn"),UInt32,
#                      (CPtr,CPtr,CPtr,Cint,Cptr), convDesc,xDesc,wDesc,ndims(x),yDims))
#     y = KnetArray(T, reverse(yDims)...)
#     yDesc = cudnnCreateTensorDescriptor(y)
#     cudnnCheck(ccall((:cudnnConvolutionForward,"libcudnn"),UInt32,
#                      (CPtr,CPtr,CPtr,CPtr,CPtr,CPtr,CPtr,UInt32,CPtr,Csize_t,CPtr,CPtr,CPtr),
#                      cudnnhandle,T[alpha],xDesc,x,wDesc,w,convDesc,algo,workSpace,workSpaceSizeInBytes,T[beta],yDesc,y))
#     cudnnDestroyTensorDescriptor(xDesc)
#     cudnnDestroyTensorDescriptor(yDesc)
#     cudnnDestroyFilterDescriptor(wDesc)
#     cudnnDestroyConvolutionDescriptor(convDesc)
#     return y
# end

# # do we need to do a special bias or is cuda12 enough?

# @primitive conv4(w,x;o...),dy  conv4back(x,dy)

# # conv4back
# # 8 descriptor functions
# # pool and poolback
