"""

    gaussian(a...; mean=0.0, std=0.01)

Return a Gaussian array with a given mean and standard deviation.  The
`a` arguments are passed to `randn`.

"""
function gaussian(a...; mean=0.0, std=0.01)
    r = randn(a...)
    T = eltype(r)
    r .* T(std) .+ T(mean)
end

"""

    xavier(a...)

Xavier initialization returns uniform random weights in the range `±sqrt(2 / (fanin +
fanout))`.  The `a` arguments are passed to `rand`.  See ([Glorot and Bengio
2010](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)) for a description.
[Caffe](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html#details)
implements this slightly differently.
[Lasagne](http://lasagne.readthedocs.org/en/latest/modules/init.html#lasagne.init.GlorotUniform)
calls it `GlorotUniform`.

"""
function xavier(a...)
    w = rand(a...)
    if ndims(w) == 1
        fanout = 1
        fanin = length(w)
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w))
        fanin = div(length(w), fanout)
    end
    s = convert(eltype(w), sqrt(2 / (fanin + fanout)))
    w = 2s .* w .- s
end


"""

    xavier_uniform(a...)

Xavier initialization returns uniform random weights in the range `±sqrt(6 / (fanin +
fanout))`.  The `a` arguments are passed to `rand`.  See ([Glorot and Bengio
2010](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)) for a description.
The function implements equation (16) of the referenced paper.

Unlike the `xavier` function, `xavier_uniform` uses the same convention as
TensorFlow and PyTorch.
"""
function xavier_uniform(a...)
    w = rand(a...)
    if ndims(w) == 1
        fanout = 1
        fanin = length(w)
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        # if a is (3,3,16,8), then there are 16 input channels and 8
        # output channels
        # fanin = 3*3*16 = (3*3*16*8) ÷ 8
        # fanout = 3*3*8 = (3*3*16*8) ÷ 16
        fanin = div(length(w),  a[end])
        fanout = div(length(w), a[end-1])
    end
    s = convert(eltype(w), sqrt(6 / (fanin + fanout)))
    w = 2s .* w .- s
end


"""

Bilinear interpolation filter weights; used for initializing deconvolution layers.

Adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py#L33

Arguments:

`T` : Data Type

`fw`: Width upscale factor

`fh`: Height upscale factor

`IN`: Number of input filters

`ON`: Number of output filters


Example usage:

w = bilinear(Float32,2,2,128,128)

"""
function bilinear(T,fw,fh,IN,ON)
    @assert fw == fh "Filter must be square"
    @assert IN == ON "Number of input and output filters must be equal"
    f=fw; N=IN;

    sz = 2*f-f%2
    center = f-0.5
    if sz%2 == 1
        center = f-1
    end
    r = range(0,stop=sz-1,length=sz)
    c = range(0,stop=sz-1,length=sz)'

    kernel = (1 .- abs.(r .- center) ./ f) .* (1 .- abs.(c .- center) ./ f)
    w = zeros(T,sz,sz,N,N);
    for i=1:N
        w[:,:,i,i] = kernel
    end
    return w
end
