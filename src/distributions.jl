function gaussian(a...; mean=0.0, std=0.01)
	return randn(a...) * std + mean;
end

function xavier(a...)
    w = rand(a...)
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in xavier")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w))
        fanin = div(length(w), fanout)
    end
    s = sqrt(2 / (fanin + fanout))
    w = 2s*w-s
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
    r = linspace(0,sz-1,sz)
    c = linspace(0,sz-1,sz)'

    kernel = (1-abs(r-center)/f)*(1-abs(c-center)/f)
    w = zeros(T,sz,sz,N,N);
    for i=1:N
        w[:,:,i,i] = kernel
    end
    return w
end