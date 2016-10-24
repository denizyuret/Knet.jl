if GPU
    import CUDNN: cudnnConvolutionForward, cudnnConvolutionBackwardFilter, cudnnConvolutionBackwardData, cudnnPoolingForward, cudnnPoolingBackward
    using CUDNN: CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, CUDNN_CONVOLUTION_FWD_ALGO_FFT, CUDNN_POOLING_MAX, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
else
    const CUDNN_CONVOLUTION = (UInt32)(0)
    const CUDNN_CROSS_CORRELATION = (UInt32)(1)
    const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = (UInt32)(0)
    const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = (UInt32)(1)
    const CUDNN_CONVOLUTION_FWD_ALGO_GEMM = (UInt32)(2)
    const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = (UInt32)(3)
    const CUDNN_CONVOLUTION_FWD_ALGO_FFT = (UInt32)(4)
    const CUDNN_POOLING_MAX = (UInt32)(0)
    const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = (UInt32)(1)
    const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = (UInt32)(2)
end

function _conv2_gemm{T}(x0::Array{T,2}, w::Array{T,2}; pad=0, stride=1, xcorr=false)
    if pad > 0
        x=zeros(eltype(x0),map(m->2pad+m,size(x0))) 
        x[pad+1:end-pad,pad+1:end-pad] = x0
    else
        x=x0
    end
    rwindow, cwindow = size(w)
    row_extend = size(x,1)-rwindow+1
    col_extend = size(x,2)-cwindow+1

    widx = [(j-1)*size(x,1)+i for i in 1:row_extend, j in 1:col_extend]
    oidx = [(j-1)*size(x,1)+i for i in 1:rwindow, j in 1:cwindow]
    destidx = [i+(j-1) for i in widx, j in oidx]
    return reshape(x[destidx]*(xcorr ? w[:] : reverse(w[:])),row_extend,col_extend)
end

function _conv2{T}(x::Array{T,2}, w::Array{T,2}; pad=0, stride=1, xcorr=false)
    max_pad = map(x->x-1-pad,size(w))
    y = conv2(x, xcorr ? rot180(w) : w)
    return y[1+max_pad[1]:stride:end-max_pad[1], 1+max_pad[2]:stride:end-max_pad[2]]
end

function cudnnConvolutionForward{T}(x::Array{T,4}, w::Array{T,4}, y::Array{T,4}; padding=0, stride=1, 
                                    upscale=1, mode=CUDNN_CONVOLUTION, cd=nothing,
                                    algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                    workSpace=0, workSpaceSizeInBytes=0, alpha=1, beta=1,im2col=1)
    # x: (W,H,C,N)
    # w: (W,H,C,K) 
    # y: (W,H,K,N) 
    fill!(y,0)
    @assert (padding==0 && stride==1 && upscale==1 && mode==CUDNN_CONVOLUTION && algorithm == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) "$((padding,stride,upscale,mode,algorithm))"
    Wx,Hx,Cx,N = size(x)
    Ww,Hw,Cw,K = size(w)
    @assert (Cx==Cw && Hx>=Hw && Wx>=Ww) "$((Wx,Hw,Ww,Hw))"

    @inbounds for n in 1:N, k in 1:K, c in 1:Cx
        y[:,:,k,n] += _conv2_gemm(x[:,:,c,n], w[:,:,c,k]; pad=padding, stride=stride, xcorr=mode!=0)
    end
    return y
end

# dw = rot180(xcorr(x,dy))
function cudnnConvolutionBackwardFilter{T}(x::Array{T,4}, dy::Array{T,4}, dw::Array{T,4}; padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION)
    # x:    (Wx,Hx,Cx,N)
    # dy:   (Wy,Hy,K,N) 
    # dw:    (Ww,Hw,Cw,K) 
    fill!(dw,0)
    @assert (padding==0&& stride==1&& upscale==1&& mode==CUDNN_CONVOLUTION)
    Wx,Hx,C,Nx = size(x)
    Wy,Hy,K,Ny = size(dy)
    @inbounds for c in 1:C, k in 1:K, n in 1:Ny
        dw[:,:,c,k] += rot180(_conv2_gemm(x[:,:,c,n], dy[:,:,k,n]; pad=padding, stride=stride, xcorr=true))
    end
    return dw
end

# dx = xcorr(dy, w, 'full')
function cudnnConvolutionBackwardData{T}(w::Array{T,4}, dy::Array{T,4}, dx::Array{T,4}; padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION)
    fill!(dx,0)
    @assert (padding==0&& stride==1&& upscale==1&& mode==CUDNN_CONVOLUTION)
    Wy,Hy,Ky,N = size(dy)
    Ww,Hw,C,Kw = size(w)
    @assert Ky==Kw
    @inbounds for n in 1:N, c in 1:C, k in 1:Kw
        t = _conv2_gemm(dy[:,:,k,n], w[:,:,c,k]; xcorr=true, pad=Ww-1)
        # t = conv2(dy[:,:,k,n], rot180(w[:,:,c,k]))
        # t = _conv2(dy[:,:,k,n], w[:,:,c,k]; pad=Ww-1, stride=stride, xcorr=true)
        dx[:,:,c,n] += t
    end
    return dx
end


function cudnnPoolingForward{T}(x::Array{T,4}, y; window=2, padding=0, stride=window, mode=CUDNN_POOLING_MAX)
    fill!(y,0)
    @assert (padding==0 &&  mode==CUDNN_POOLING_MAX &&  stride==window)
    # x: (W,H,C,N)
    Wx,Hx,C,Nx = size(x);
    Wy,Hy,K,Ny = size(y);
    @assert (Nx == Ny && C==K)
    @inbounds for n in 1:Nx, c in 1:C, j in 1:stride:Hx, i in 1:stride:Wx
        iy, jy = div(i,stride)+1, div(j,stride)+1
        hx_end = (j+window-1 >= Hx) ? Hx : j+window-1
        wx_end = (i+window-1 >= Wx) ? Wx : i+window-1
        y[iy,jy,c,n] = maximum(x[i:wx_end,j:hx_end,c,n])
    end
    return y
end

function cudnnPoolingBackward{T}(y::Array{T,4}, dy::Array{T,4}, x::Array{T,4}, dx::Array{T,4}; window=2, padding=0, stride=1, mode=CUDNN_POOLING_MAX)
    fill!(dx,0)
    @assert (padding==0 && stride==window && mode==CUDNN_POOLING_MAX)
    # x: (W,H,C,N)
    Wx,Hx,C,Nx = size(x);
    Wy,Hy,K,Ny = size(y);
    @assert (Nx == Ny && C==K)
    @inbounds for n in 1:Nx, c in 1:C, j in 1:stride:Hx, i in 1:stride:Wx
        iy, jy = div(i,stride)+1, div(j,stride)+1
        hx_end = (j+window-1 >= Hx) ? Hx : j+window-1
        wx_end = (i+window-1 >= Wx) ? Wx : i+window-1
        a = x[i:wx_end,j:hx_end,c,n]
        di,dj = ind2sub(a,indmax(a))
        dx[i+di-1,j+dj-1,c,n] += dy[iy,jy,c,n]
    end
    return dx
end

function cudnnGetConvolutionNdForwardOutputDim{T}(x::Array{T,4}, w::Array{T,4}; padding=padding,stride=stride)
    Wx,Hx,Cx,N = size(x)
    Ww,Hw,Cw,K = size(w)
    @assert Cx==Cw
    Wy,Hy = floor(Int, 1 + (Int[Wx,Hx] + 2*padding - Int[Ww,Hw]) / stride)
    return (Wy,Hy,K,N)
end

function cudnnGetPoolingNdForwardOutputDim{T}(x::Array{T,4}; window=2, padding=0, stride=1, mode=CUDNN_POOLING_MAX)
    # @assert padding==0 && stride==1 && mode==CUDNN_POOLING_MAX
    dims = [size(x)...]
    # (mode, pdims, window, padding, stride) = cudnnGetPoolingNdDescriptor(pd)
    for i=1:length(dims)-2
        # dims[i] = 1 + floor((dims[i] + 2*padding - window) / stride)
        dims[i] = 1 + ceil((dims[i] + 2*padding - window) / stride)
    end
    tuple(dims...)
end
