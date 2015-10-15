### TODO: Rethink the CUDNN interface

using CUDNN: cudnnConvolutionDescriptor_t, cudnnCreateConvolutionDescriptor, cudnnSetConvolutionNdDescriptor_v3, cudnnDestroyConvolutionDescriptor, cudnnHandle, cptr, TD, FD, cudnnDataType
using CUDNN: CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
using CUDNN: cudnnGetConvolutionNdForwardOutputDim, cudnnGetConvolutionForwardWorkspaceSize
using CUDNN: cudnnConvolutionBackwardFilter_v3, cudnnConvolutionBackwardData_v3

function cudnnConvolutionForward_v4(src, filter, dest;
                                    handle=cudnnHandle, alpha=1.0, beta=0.0, 
                                    algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                                    workSpace=C_NULL, workSpaceSizeInBytes=0,
                                    padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION)
    @assert eltype(filter) == eltype(src)
    cd = cudnnConvolutionDescriptor_v4(ndims(src), eltype(src), padding, stride, upscale, mode)
    osize = cudnnGetConvolutionNdForwardOutputDim(src,filter;convDesc=cd)
    (dest == nothing) && (dest = CudaArray(eltype(src), osize))
    @assert osize == size(dest)
    @assert eltype(dest) == eltype(src)
    wsize = cudnnGetConvolutionForwardWorkspaceSize(src, filter, dest; algorithm=algorithm)
    if ((wsize > 0) && (workSpace == C_NULL || workSpaceSizeInBytes < wsize))
        workSpaceSizeInBytes = wsize
        ws = CudaArray(Int8, workSpaceSizeInBytes)
    else
        ws = workSpace
    end
    cudnnConvolutionForward(handle,
                            cptr(alpha,src),TD(src),src,
                            FD(filter),filter,
                            cd,algorithm,ws,workSpaceSizeInBytes,
                            cptr(beta,dest),TD(dest),dest)
    free(cd)
    ws === workSpace || free(ws)
    return dest
end

# I am guessing if y=w*x+b going forward, the arguments below
# correspond to src=x, diff=dy, grad=dw.
function cudnnConvolutionBackwardFilter_v4(src, diff, grad;
                                           handle=cudnnHandle, alpha=1.0, beta=0.0, 
                                           algo=CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                           workSpace=C_NULL, workSpaceSizeInBytes=0,
                                           padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION)
    cd = cudnnConvolutionDescriptor_v4(ndims(src), eltype(src), padding, stride, upscale, mode)
    cudnnConvolutionBackwardFilter_v3(handle,
                                      cptr(alpha,src),TD(src),src,
                                      TD(diff),diff,cd,
                                      algo, workSpace, workSpaceSizeInBytes,
                                      cptr(beta,grad),FD(grad),grad)
    free(cd)
    return grad
end

# I am guessing if y=w*x+b going forward, the arguments below
# correspond to filter=w, diff=dy, grad=dx.
function cudnnConvolutionBackwardData_v4(filter::AbstractCudaArray, diff::AbstractCudaArray, grad::AbstractCudaArray;
                                         handle=cudnnHandle, alpha=1.0, beta=0.0, 
                                         algo=CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                         workSpace=C_NULL, workSpaceSizeInBytes=0,
                                         padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION)
    cd = cudnnConvolutionDescriptor_v4(ndims(filter), eltype(filter), padding, stride, upscale, mode)
    cudnnConvolutionBackwardData_v3(handle,cptr(alpha,diff),
                                    FD(filter),filter,
                                    TD(diff),diff,cd,
                                    algo, workSpace, workSpaceSizeInBytes,
                                    cptr(beta,grad),TD(grad),grad)
    free(cd)
    return grad
end


# Decided not to keep the cd in the Pool structure: not serializable.
function cudnnConvolutionDescriptor_v4(xdims, xtype, padding, stride, upscale, mode)
    cd = cudnnConvolutionDescriptor_t[0]
    cudnnCreateConvolutionDescriptor(cd)
    nd = xdims-2
    cudnnSetConvolutionNdDescriptor_v3(cd[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode,cudnnDataType(xtype))
    return cd[1]
end

function cdsize(w, nd)
    isa(w,Integer) ? Cint[fill(w,nd)...] :
    length(w)!=nd ? error("Dimension mismatch") :
    Cint[reverse(w)...]
end

CUDArt.free(cd::cudnnConvolutionDescriptor_t)=cudnnDestroyConvolutionDescriptor(cd)

### TODO: THIS INTERFACE SHOULD GO INTO CUDNN:

using CUDNN: cudnnPoolingDescriptor_t, cudnnCreatePoolingDescriptor, cudnnSetPoolingNdDescriptor, cudnnDestroyPoolingDescriptor, cudnnHandle, cptr, TD, cudnnPoolingMode_t, cudnnGetPoolingNdDescriptor

function cudnnPoolingForward_v4(src, dest; 
                                window=2, padding=0, stride=window, mode=CUDNN_POOLING_MAX, 
                                handle=cudnnHandle, alpha=1.0, beta=0.0)
    pd = cudnnPoolingDescriptor_v4(ndims(src)-2, window, padding, stride, mode)
    cudnnPoolingForward(handle, pd, 
                        cptr(alpha,src), TD(src), src,
                        cptr(beta,dest), TD(dest), dest)
    free(pd)
    return dest
end

function cudnnPoolingBackward_v4(src, srcDiff, dest, destDiff; 
                                 window=2, padding=0, stride=window, mode=CUDNN_POOLING_MAX, 
                                 handle=cudnnHandle, alpha=1.0, beta=0.0)
    pd = cudnnPoolingDescriptor_v4(ndims(src)-2, window, padding, stride, mode)
    cudnnPoolingBackward(handle, pd, 
                         cptr(alpha,src), TD(src), src, 
                         TD(srcDiff), srcDiff, 
                         TD(dest), dest,
                         cptr(beta,destDiff), TD(destDiff), destDiff)
    free(pd)
    return destDiff
end

# Decided not to keep the pd in the Pool structure: not serializable.
function cudnnPoolingDescriptor_v4(nd, window, padding, stride, mode)
    pd = cudnnPoolingDescriptor_t[0]
    cudnnCreatePoolingDescriptor(pd)
    cudnnSetPoolingNdDescriptor(pd[1],mode,nd,pdsize(window,nd),pdsize(padding,nd),pdsize(stride,nd))
    return pd[1]
end

function cudnnGetPoolingNdDescriptor_v4(pd::cudnnPoolingDescriptor_t, nd)
    m = cudnnPoolingMode_t[0]
    n = Cint[0]
    s = Array(Cint, nd)
    p = Array(Cint, nd)
    t = Array(Cint, nd)
    cudnnGetPoolingNdDescriptor(pd, nd, m, n, s, p, t)
    inttuple(x)=tuple(Int[x...]...)
    (m[1], n[1], inttuple(s), inttuple(p), inttuple(t))
end

CUDArt.free(pd::cudnnPoolingDescriptor_t)=cudnnDestroyPoolingDescriptor(pd)

pdsize(w, nd)=Cint[reverse(psize(w,nd))...]

psize(w, nd)=(isa(w,Integer)  ? fill(w,nd) : length(w) != nd ? error("Dimension mismatch") : w)

