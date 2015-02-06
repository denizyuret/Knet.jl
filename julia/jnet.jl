module Jnet

include("jnet_util.jl")
include("jnet_test.jl")

type Layer{t}
    fforw::Function	# y=forw(x)=fforw(w*x+b)
    fback::Function     # dx=back(dy)=w'*fback(dy)
    w::Mat{t}           # weight matrix
    b::Vec{t}           # bias vector

    dw::Mat{t}          # gradient wrt weight matrix
    dw1::Mat{t}         # moving average of gradients for momentum
    dw2::Mat{t}         # sum of squared gradients for adagrad

    db::Vec{t}          # gradient wrt bias vector
    db1::Vec{t}         # moving average of gradients for momentum
    db2::Vec{t}         # sum of squared gradients for adagrad

    x::Mat{t}           # last input
    y::Mat{t}           # last output
    dx::Mat{t}          # gradient wrt input
    dy::Mat{t}          # gradient wrt output
    xmask::Mat{t}       # input mask for dropout
    xones::Vec{t}       # vector of ones for bias calculation

    learningRate::t     # learning rate
    momentum::t         # momentum
    adagrad::Bool       # boolean indicating adagrad trick
    nesterov::Bool      # boolean indicating nesterov trick
    dropout::t          # probability of dropping inputs
    maxnorm::t          # parameter for maxnorm regularization
    L1::t               # parameter for L1 regularization
    L2::t               # parameter for L2 regularization

    ## Defaults:

    Layer(;_v=Array(t,0), _m=Array(t,0,0), _z=zero(t), _f=(x->x),
	  fforw=_f, fback=_f, w=_m, b=_v, dw=_m, dw1=_m,
	  dw2=_m, db=_v, db1=_v, db2=_v, x=_m, y=_m, dx=_m, dy=_m,
	  xmask=_m, xones=_v, learningRate=_z, momentum=_z,
	  adagrad=false, nesterov=false, dropout=_z, maxnorm=_z,
	  L1=_z, L2=_z) =
    new(fforw, fback, w, b, dw, dw1, dw2, db, db1, db2, x, y, dx, dy,
	xmask, xones, learningRate, momentum, adagrad, nesterov,
	dropout, maxnorm, L1, L2)
end # Layer


### Basic layer functions

function forw{t}(l::Layer{t}, x::Mat{t})
    l.x = initforw(l, x)                                # alloc y,xones
    gemm!('N', 'N', one(t), l.w, l.x, zero(t), l.y) 	# y=w*x
    if !isempty(l.b)
	ger!(one(t), l.b, l.xones, l.y)                 # y=y+b
    end
    l.fforw(l.y)                                        # y=fforw(y)
    return l.y
end

function back{t}(l::Layer{t}, dy::Mat{t}, dx::Bool=true)
    l.dy = initback(l, dy, dx)		
    l.fback(l.dy, l.y)	     # this will overwrite l.dy and l.y!
    gemm!('N', 'T', one(t), l.dy, l.x, zero(t), l.dw)	# dw=dy*x'
    if !isempty(l.b)
	gemv!('N', one(t), l.dy, l.xones, zero(t), l.db) # db=sum(dy,2)
    end
    if (dx) # dx is optional because it is expensive and unnecessary for input layer
        gemm!('T', 'N', one(t), l.w, l.dy, zero(t), l.dx) # dx=w'*dy
        return l.dx
    end
end

function x = drop(l, x)		# TODO: before or after copy to l.x?
				# probably after, we don't want to overwrite training data, so we should always copy?
    # Drop each element of the input x with probability l.dropout.
    l.xmask = (l.randlike(x) > l.dropout);  # TODO: find or implement rand for gpu, rand! ?  also need >
    x(:) = x .* l.xmask * (1/(1-l.dropout));  # TODO: blas vector multiply? = sbmv
				# Do the whole thing as a kernel like reluforw.
end

function update{t}(l::Layer{t})
    if l.L1
        l.dw(:) = l.dw + l.L1 * sign(l.w); # axpy and sign
    end
    if l.L2
        l.dw(:) = l.dw + l.L2 * l.w; # axpy
    end
    if l.adagrad
        if ~isempty(l.dw2)
            l.dw2(:) = l.dw .* l.dw + l.dw2;
        else
            l.dw2 = l.dw .* l.dw;
        end
        l.dw(:) = l.dw ./ (1e-8 + sqrt(l.dw2));	# sqrt? inv?
    end
    if ~isempty(l.learningRate)
        l.dw(:) = l.learningRate * l.dw; # scale?
    end
    if l.momentum
        if ~isempty(l.dw1)
            l.dw1(:) = l.dw + l.momentum * l.dw1; # axpy?
        else
            l.dw1 = l.dw;
        end
        if l.nesterov
            l.dw(:) = l.dw + l.momentum * l.dw1;
        else
            l.dw(:) = l.dw1;
        end
    end

    l.w(:) = l.w - l.dw;	# axpy

    if l.maxnorm
        norms = sqrt(sum(l.w.^2, 2)); # TODO:gpu version
        if any(norms > l.maxnorm)
            scale = min(l.maxnorm ./ norms, 1);
            l.w(:) = bsxfun(@times, l.w, scale);
        end
    end
end



### Layer types

relu{t}(w0::Mat{t},b0::Vec{t}) = Layer{t}(fforw=reluforw, fback=reluback, w=w0, b=b0)
soft{t}(w0::Mat{t},b0::Vec{t}) = Layer{t}(fforw=softforw, fback=softback, w=w0, b=b0)

function reluforw{t}(y::Mat{t})
    for i=1:length(y)
        if (y[i] < zero(t))
            y[i] = zero(t)
        end
    end
    return y
end

function reluback{t}(dy::Mat{t}, y::Mat{t})
    for i=1:length(dy)
        if (y[i] <= zero(t))
            dy[i] = zero(t)
        end
    end
end

softforw(x)=x

function softback{t}(dy::Mat{t}, y::Mat{t})
    # we do softmax here instead of in forw
    # overwriting y from unnormalized log probabilities to normalized probabilities
    # NumericExtensions.softmax!(y,y,1) allocates unnecessary memory
    # dy is a 0-1 matrix of correct answers
    # will overwrite it with the gradient
    # TODO: is this a good interface?
    # TODO: other types of final layers, losses?

    for j=1:size(y,2)
        ymax = y[1,j]
        for i=2:size(y,1)
            if (y[i,j] > ymax)
                ymax = y[i,j]
            end
        end
        ysum = zero(t)
        for i=1:size(y,1)
            y[i,j] = exp(y[i,j] - ymax)
            ysum += y[i,j]
        end
        for i=1:size(y,1)
            y[i,j] /= ysum
            dy[i,j] = (y[i,j] - dy[i,j]) / size(y,2)
        end
    end
end


### A Net is just an array of Layers

typealias Net{t} Array{Layer{t},1}

function forw{t}(net::Net{t}, x::Mat{t})
    x = initforw(net, x)
    for i=1:length(net)
        x = forw(net[i], x)
    end
    return x
end

function back{t}(net::Net{t}, dy::Mat{t})
    dy = initback(net, dy)
    for i=length(net):-1:2
        dy = back(net[i], dy)
    end
    # No need to compute the last dx
    back(net[1], dy, false)
end


### Batch processing:

function forw{t}(net::Net{t}, x::Mat{t}, batch::Int)
    xrows,xcols = size(x)
    yrows,ycols = size(net[end].w, 1), xcols
    y = zeros(t, yrows, ycols)
    info("forw:Alloc(y)=$(mysizeof(y))")
    for b=1:batch:xcols
        e = b + batch - 1
        if (e > xcols) e = xcols; end
        y[:,b:e] = to_host(forw(net, sub(x,1:xrows,b:e)))
    end
    return y
end

function forwback{t}(net::Net{t}, x::Mat{t}, labels::Vec{t}, batch::Int)
    xrows,xcols = size(x)
    yrows,ycols = size(net[end].w, 1),xcols
    y = zeros(t, yrows, ycols)
    info("forwback:Alloc(y)=$(mysizeof(y))")
    for i=1:length(labels)
        y[labels[i],i] = 1
    end
    for b=1:batch:xcols
        e = b + batch - 1
        if (e > xcols) e = xcols; end
        forw(net, sub(x,1:xrows,b:e))
        back(net, sub(y,1:yrows,b:e))
    end
end


### Memory management for forw and back
# This code is ugly because:
# 1. Need to free CudaArrays rather than relying on gc()
# 2. With getindex, ones, etc. missing from CUDArt, not possible to write generic code

function initmat{t}(l::Layer{t}, n::Symbol, dims::Dims, init::t=zero(t))
    if (size(l.(n)) != dims)
	free(l.(n))
	l.(n) = similar(l.w, t, dims)
	fill!(l.(n), init)
        info("initmat($(n) $(dims))=$(mysizeof(l.(n)))")
    end
end

function initforw{t}(net::Net{t}, x::Mat{t})
    l = net[1]
    if (isa(l.w, CudaArray))
	initmat(l, :x, size(x))
	copy!(l.x, x)
	x = l.x
    end
    return x
end

function initback{t}(net::Net{t}, dy::Mat{t})
    l = net[length(net)]
    if (isa(l.w, CudaArray))
	initmat(l, :dy, size(dy))
	copy!(l.dy, dy)
	dy = l.dy
    end
    return dy
end

function initforw{t}(l::Layer{t}, x::Mat{t})
    if (isempty(l.w))
        error("Please initialize w")
    end
    rows = size(l.w,1)
    cols = size(x,2)
    initmat(l, :y, (rows,cols))
    if (!isempty(l.b))
	initmat(l, :xones, (cols,), one(t))
    end
    return x
end

function initback{t}(l::Layer{t}, dy::Mat{t}, dx::Bool)
    initmat(l, :dw, size(l.w))
    if (!isempty(l.b))
	initmat(l, :db, size(l.b))
    end
    if (dx) 
	initmat(l, :dx, size(l.x))
    end
    return dy
end

export Layer, forw, forwback, relu, soft

end # module Jnet
