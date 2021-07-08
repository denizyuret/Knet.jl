####
#### Sampling
####
mutable struct Sampling{T<:Function} <: Layer
    options::NamedTuple
end

Sampling{T}(;window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0, alpha=1) where T <: Function =
    Sampling{T}((window=window, padding=padding, stride=stride, mode=mode, maxpoolingNanOpt=maxpoolingNanOpt, alpha=alpha))

@inline (m::Sampling{typeof(pool)})(x)   =  pool(x;m.options...)
@inline (m::Sampling{typeof(unpool)})(x) =  unpool(x;m.options...)

Base.show(io::IO,m::Sampling{typeof(pool)})   = print(io,"Pool",m.options)
Base.show(io::IO,m::Sampling{typeof(unpool)}) = print(io,"UnPool",m.options)
"""
    Pool(keywords...)
Creates a pooling layer, `Sampling{typeof(pool)}`, according to given keyword arguments.

    (::Sampling{typeof(pool)})(x)
Compute pooling of input values (i.e., the maximum or average of several adjacent values)
to produce an output with smaller height and/or width.

Currently 4 or 5 dimensional KnetArrays with Float32 or Float64 entries are
supported. If x has dimensions (X1,X2,...,I,N), the result y will have dimensions
(Y1,Y2,...,I,N) where

Yi=1+floor((Xi+2*padding[i]-window[i])/stride[i])

Here I is the number of input channels, N is the number of instances, and Xi,Yi
are spatial dimensions. window, padding and stride are keyword arguments that can
be specified as a single number (in which case they apply to all dimensions), or
an array/tuple with entries for each spatial dimension.

Keywords:

* window=2: the pooling window size for each dimension.
* padding=0: the number of extra zeros implicitly concatenated at the
start and at the end of each dimension.
* stride=window: the number of elements to slide to reach the next pooling
window.
* mode=0: 0 for max, 1 for average including padded values, 2 for average
excluding padded values.
* maxpoolingNanOpt=0: Nan numbers are not propagated if 0, they are
propagated if 1.
* alpha=1: can be used to scale the result.

"""
const Pool = Sampling{typeof(pool)}

"""
    UnPool(kwargs...)
    (::Sampling{typeof(unpool)})(x)

    Reverse of pooling. It has same kwargs with Pool. see `Pool`

    x == pool(unpool(x;o...); o...)
"""
const UnPool = Sampling{typeof(unpool)}
####
#### Filtering
####
mutable struct Filtering{T<:Function,P,A<:ActOrNothing,V<:Bias} <: Layer
    weight::P
    bias::V
    activation::A
    options::NamedTuple
end

function Filtering{T}(;height::Integer, width::Integer, inout::Pair=1=>1,
                       activation::ActOrNothing=NonAct(),
                       winit=xavier, binit=zeros,
                       atype=arrtype,
                       opts...) where T <: Function

    wsize = T===typeof(conv4) ? inout : reverse(inout)
    w = param(height,width,wsize...; init=winit, atype=atype)
    b = binit !== nothing ? Bias(1,1,inout[2],1; init=binit, atype=atype) : Bias(nothing)
    Filtering{T}(w, b, activation; opts...)

end

Filtering{T}(w, b, activation; stride=1, padding=0, mode=0, dilation=1, alpha=1) where T =
    Filtering{T,typeof(w),typeof(activation),typeof(b)}(w, b, activation, (stride=stride, dilation=dilation, mode=mode, alpha=alpha, padding=padding))

@inline (m::Filtering{typeof(conv4)})(x) =
     unmake4D(postConv(m, conv4(m.weight, make4D(x); m.options...)), ndims(x))

Base.show(io::IO,m::Filtering{typeof(conv4),P,A,V}) where {P,A,V} =
    print(io,"Conv{$P,$A,$V}",m.options)

@inline (m::Filtering{typeof(deconv4)})(x) =
    unmake4D(postConv(m, deconv4(m.weight, make4D(x); m.options...)), ndims(x))

Base.show(io::IO,m::Filtering{typeof(deconv4),P,A,V}) where {P,A,V} =
    print(io,"DeConv{$P,$A,$V}",m.options)

"""
    Conv(;height=filterHeight, width=filterWidth, inout = 1 => 1, kwargs...)

Creates and convolutional layer `Filtering{typeof(conv4)}` according to given filter dimensions.

    (m::Filtering{typeof(conv4)})(x) #forward run

If `m.w` has dimensions `(W1,W2,...,I,O)` and
`x` has dimensions `(X1,X2,...,I,N)`, the result `y` will have
dimensions `(Y1,Y2,...,O,N)` where

    Yi=1+floor((Xi+2*padding[i]-Wi)/stride[i])

Here `I` is the number of input channels, `O` is the number of output channels, `N` is the number of instances,
and `Wi,Xi,Yi` are spatial dimensions. `padding` and `stride` are
keyword arguments that can be specified as a single number (in which case they apply to all dimensions),
or an tuple with entries for each spatial dimension.

# Keywords
* `inout=input_channels => output_channels`
* `activation=NonAct()`: nonlinear function applied after convolution, default is identity.
* `pool=nothing`: Pooling layer or window size of pooling
* `winit=xavier`: weight initialization distribution
* `bias=zeros`: bias initialization distribution
* `padding=0`: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.
* `stride=1`: the number of elements to slide to reach the next filtering window.
* `dilation=1`: dilation factor for each dimension.
* `mode=0`: 0 for convolution and 1 for cross-correlation.
* `alpha=1`: can be used to scale the result.
* `handle`: handle to a previously created cuDNN context. Defaults to a Knet allocated handle.

"""
const Conv = Filtering{typeof(conv4)}

"""
    DeConv(;height=filterHeight, width=filterWidth, inout=1=>1, kwargs...)

Creates and deconvolutional layer `Filtering{typeof(deconv4)}`  according to given filter dimensions.


    (m::Filtering{typeof(deconv4)})(x) #forward run

If `m.w` has dimensions `(W1,W2,...,I,O)` and
`x` has dimensions `(X1,X2,...,I,N)`, the result `y` will have
dimensions `(Y1,Y2,...,O,N)` where

    Yi = Wi+stride[i](Xi-1)-2padding[i]

Here `I` is the number of input channels, `O` is the number of output channels, `N` is the number of instances,
and `Wi,Xi,Yi` are spatial dimensions. `padding` and `stride` are
keyword arguments that can be specified as a single number (in which case they apply to all dimensions),
or an tuple with entries for each spatial dimension.

# Keywords
* `inout=input_channels => output_channels`
* `activation=identity`: nonlinear function applied after convolution
* `unpool=nothing`: Unpooling layer or window size of unpooling
* `winit=xavier`: weight initialization distribution
* `bias=zeros`: bias initialization distribution
* `padding=0`: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.
* `stride=1`: the number of elements to slide to reach the next filtering window.
* `dilation=1`: dilation factor for each dimension.
* `mode=0`: 0 for convolution and 1 for cross-correlation.
* `alpha=1`: can be used to scale the result.
* `handle`: handle to a previously created cuDNN context. Defaults to a Knet allocated handle.
"""
const DeConv = Filtering{typeof(deconv4)}

###
### Utils
###
@inline function make4D(x)
    n = ndims(x)
    @assert n < 5 "filtering layers currently supports 4 dimensional arrays"
    n == 4 ? x : reshape(x,size(x)...,ntuple(x->1, 4-n)...)
end

@inline unmake4D(y,dims::Int) = dims>3 ? y : reshape(y,size(y)[1:dims])

@inline postConv(m::Filtering{<:Any,<:Any,<:Activation,<:Bias}, y) = m.activation(m.bias(y))

@inline postConv(m::Filtering{<:Any,<:Any,<:Activation,<:Nothing}, y) = m.activation(y)

@inline postConv(m::Filtering{<:Any,<:Any,<:Nothing,<:Bias}, y) = m.bias(y)
