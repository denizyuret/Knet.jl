#Generate a Gaussian distribution with specified mean and standard deviation
function gaussian(a...; mean=0.0, std=0.01)
	return randn(a...) * std + mean;
end

#Taken from "examples/lenet.jl"
function xavier(a...)
    w = rand(a...)
     # The old implementation was not right for fully connected layers:
     # (fanin = length(y) / (size(y)[end]); scale = sqrt(3 / fanin); axpb!(rand!(y); a=2*scale, b=-scale)) :
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in xavier")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w)) # Caffe disagrees: http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html#details
        fanin = div(length(w), fanout)
    end
    # See: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    s = sqrt(2 / (fanin + fanout))
    w = 2s*w-s
end

"""

Used for initializing deconvolution layers.

Adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py#L33

Example usage:

w = bilinear(Float32,2,2,2,2)

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