using AutoGrad: AutoGrad, @primitive1, value
import CUDA.CUDNN: cudnnConvolutionForwardAD
using CUDA.CUDNN:
    CUDNN_ACTIVATION_IDENTITY,
    CUDNN_ACTIVATION_RELU,
    CUDNN_NOT_PROPAGATE_NAN,
    cudnnActivationBackward,
    cudnnActivationDescriptor,
    cudnnConvolutionBackwardBias,
    cudnnConvolutionBackwardData,
    cudnnConvolutionBackwardFilter,
    cudnnConvolutionBwdDataAlgoPerf,
    cudnnConvolutionBwdFilterAlgoPerf,
    scalingParameter,
    with_workspace,
    handle
  

# Define gradients
@primitive1(
    (cudnnConvolutionForwardAD(w, x, bias, z; y, activation, convDesc, wDesc, xDesc, yDesc, zDesc, biasDesc, alpha, beta, dw, dx, dz, dbias, dready),_dy,_y),
    (dready[] || cudnnConvolutionBackward(_dy, _y, w, x, bias, z; y, activation, convDesc, wDesc, xDesc, yDesc, zDesc, biasDesc, alpha, beta, dw, dx, dz, dbias, dready); dw[]),
    (dready[] || cudnnConvolutionBackward(_dy, _y, w, x, bias, z; y, activation, convDesc, wDesc, xDesc, yDesc, zDesc, biasDesc, alpha, beta, dw, dx, dz, dbias, dready); dx[]),
    (dready[] || cudnnConvolutionBackward(_dy, _y, w, x, bias, z; y, activation, convDesc, wDesc, xDesc, yDesc, zDesc, biasDesc, alpha, beta, dw, dx, dz, dbias, dready); dbias[]),
    (dready[] || cudnnConvolutionBackward(_dy, _y, w, x, bias, z; y, activation, convDesc, wDesc, xDesc, yDesc, zDesc, biasDesc, alpha, beta, dw, dx, dz, dbias, dready); dz[]))


function cudnnConvolutionBackward(_dy, _y, w, x, bias, z; y, activation, convDesc, wDesc, xDesc, yDesc, zDesc, biasDesc, alpha, beta, dw, dx, dz, dbias, dready)
    # Make sure backward gets called only once
    dready[] && return
    dready[] = true
    # Read all relevant inputs
    (dy, y, w, x, bias, z) = value.((_dy, _y, w, x, bias, z))
    # Allocate gradient buffers if necessary
    if dw[] === nothing; dw[] = similar(w); end; @assert issimilar(w, dw[])
    if dx[] === nothing; dx[] = similar(x); end; @assert issimilar(x, dx[])
    if bias === nothing
        dbias[] = nothing
    elseif dbias[] === nothing
        dbias[] = similar(bias)
    end
    @assert issimilar(bias, dbias[])
    # Calculate pre-activation gradient if necessary, use dz[] for storage
    @assert isa(z,Nothing) || issimilar(z, y)
    if activation !== CUDNN_ACTIVATION_IDENTITY
        if dz[] === nothing; dz[] = similar(y); end; @assert issimilar(y, dz[])
        actback!(dz[],y,dy,y,activation)
        dy = dz[]
    end
    beta0, alpha1 = (a->scalingParameter(eltype(x),a)).((0,1))
    p = cudnnConvolutionBwdFilterAlgoPerf(xDesc, x, yDesc, dy, convDesc, wDesc, dw[]);
    with_workspace(p.memory) do workspace
        cudnnConvolutionBackwardFilter(handle(), alpha, xDesc, x, yDesc, dy, convDesc, p.algo, workspace, sizeof(workspace), beta0, wDesc, dw[])
    end
    p = cudnnConvolutionBwdDataAlgoPerf(wDesc, w, yDesc, dy, convDesc, xDesc, dx[]);
    with_workspace(p.memory) do workspace
        cudnnConvolutionBackwardData(handle(), alpha, wDesc, w, yDesc, dy, convDesc, p.algo, workspace, sizeof(workspace), beta0, xDesc, dx[])
    end
    if bias !== nothing
        cudnnConvolutionBackwardBias(handle(), alpha1, yDesc, dy, beta0, biasDesc, dbias[])
    end
    if beta[] == 0 || z === nothing
        dz[] = nothing
    elseif dz[] === nothing
        dz[] = beta[] .* dy
    else
        dz[] .= beta[] .* dy
    end
end


function actback!(dx,x,dy,y,activation)
    activationDesc = cudnnActivationDescriptor(activation, CUDNN_NOT_PROPAGATE_NAN, Cfloat(1))
    xDesc, yDesc = cudnnTensorDescriptor.((x,y))
    alpha, beta = (x->scalingParameter(eltype(y),x)).((1,0))
    cudnnActivationBackward(handle(), activationDesc, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx)
    return dx
end

