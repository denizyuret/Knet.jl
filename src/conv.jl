# ConvLayer: It has similar fields to a regular (fully connected)
# Layer.  All fields are optional and specified using keyword
# arguments in the constructor.  The operations with undefined fields
# will not be performed.  Differences from the fully connected Layer:
#
# w, x, etc. are tensors rather than matrices.
# f is a CUDNN.cudnnActivationMode_t rather than a function.
# cd and pd are ConvolutionDescriptor and PoolingDescriptor.  
# ws is convolution workspace and wssb is its size in bytes.

type ConvLayer <: AbstractLayer
    w; b; f; fx; dw; db; pw; pb; x; y; z; dx; dy; dz; dropout; xdrop; 
    cd::ConvolutionDescriptor; pd::PoolingDescriptor; ws::CudaArray; wssb::Int
    ConvLayer(;a...)=setparam!(new();a...)
end

function forw(l::ConvLayer, x, apply_fx=true)
    initforw(l, x)
    isdefined(l,:fx) && apply_fx && l.fx(l,l.x)
    isdefined(l,:w) ? cudnnConvolutionForward(l.x, l.w, l.y; convDesc=l.cd, workSpace=l.ws, workSpaceSizeInBytes=l.wssb) : (l.y = l.x)
    isdefined(l,:b) && cudnnAddTensor(l.b, l.y)
    # TODO: f is a function in Layer and an id in ConvLayer, fix it.
    isdefined(l,:f) && cudnnActivationForward(l.y; mode=l.f)
    isdefined(l,:pd) ? cudnnPoolingForward(l.pd, l.y, l.z) : (l.z = l.y)
    return l.z
end

function initforw(l::ConvLayer, x)
    l.x = x
    isdefined(l,:w) && !isdefined(l,:cd) && (l.cd = CUDNN.defaultConvolutionDescriptor)
    isdefined(l,:w) && chksize(l, :y, l.w, cudnnGetConvolutionNdForwardOutputDim(l.x, l.w; convDesc=l.cd))
    isdefined(l,:b) && (@assert size(l.b) == (1,1,size(l.y,3),1))
    isdefined(l,:pd) && chksize(l, :z, l.y, cudnnGetPoolingNdForwardOutputDim(l.pd, l.y))
    wssb = cudnnGetConvolutionForwardWorkspaceSize(x, l.w, l.y; convDesc=l.cd)
    if wssb > 0
        (isdefined(l,:wssb) && (l.wssb >= wssb)) || (l.wssb = wssb)
        (isdefined(l,:ws) && (length(l.ws) * sizeof(eltype(l.ws)) >= l.wssb)) || (l.ws = CudaArray(Int8, l.wssb))
    end
end

function back(l::ConvLayer, dz, return_dx=true)
    initback(l, dz, return_dx);
    isdefined(l,:pd) ? cudnnPoolingBackward(l.pd, l.z, l.dz, l.y, l.dy) : (l.dy = l.dz)
    # TODO: make sure dest=l.y is ok in cudnnActivationBackward
    isdefined(l,:f) && cudnnActivationBackward(l.y, l.dy, l.y; mode=l.f)  # overwrites l.dy
    isdefined(l,:b) && cudnnConvolutionBackwardBias(l.dy, l.db)
    isdefined(l,:w) && cudnnConvolutionBackwardFilter(l.x, l.dy, l.dw; convDesc=l.cd)
    return_dx || return
    isdefined(l,:w) ? cudnnConvolutionBackwardData(l.w, l.dy, l.dx; convDesc=l.cd) : (l.dx = l.dy)
    # TODO: make sure fx works here with tensors
    isdefined(l,:fx) && l.fx(l,l.x,l.dx)
    return l.dx
end

function initback(l::ConvLayer, dz, return_dx)
    l.dz = tensorwithdims(dz, size(l.z))
    isdefined(l,:pd) && chksize(l, :dy, l.y)
    isdefined(l,:b) && chksize(l, :db, l.b)
    isdefined(l,:w) && chksize(l, :dw, l.w)
    return_dx && isdefined(l,:w) && chksize(l, :dx, l.x)
end

# Check and fix size in case dz is a 2D (Cuda)Array coming from a
# fully connected layer.
tensorwithdims(a, dims)=(isa(a, Tensor) && (size(a) == dims) ? a : Tensor(reinterpret(eltype(a), a, dims)))
