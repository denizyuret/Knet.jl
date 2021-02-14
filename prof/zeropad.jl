using Knet.KnetArrays: KnetArray
using CUDA: CuArray
using AutoGrad: Value
const GPUVal{T,N} = Union{KnetArray{T,N},CuArray{T,N},Value{KnetArray{T,N}},Value{CuArray{T,N}}}


# cut-and-paste method: see Knet/prof/zeropad for alternatives
# cpu=147ms (x=(56,56,128,128), p=((0,1),(0,1)))

function zeropad1(x, p; y=nothing, fillzero=true)
    p = padtuple(p, size(x))
    n = ndims(x)
    d = ntuple(i->(i < n-1 ? size(x,i)+sum(p[i]) : size(x,i)), n)
    c = ntuple(i->(i < n-1 ? (p[i][1]+1:p[i][1]+size(x,i)) : Colon()), n)
    y === nothing ? y = similar(x,d) : @assert typeof(y)===typeof(x) && size(y)==d
    fillzero && fill!(y, 0)
    y[c...] .= x
    return y
end

# concatenate: 185ms (x=(56,56,128,128), p=((0,1),(0,1)))
function zeropad2(x, p)
    pad(d) = fill!(similar(x, d), 0)
    p = padtuple(p, size(x))
    for i in 1:length(p)
        pad1 = pad(ntuple(j->(j==i ? p[j][1] : size(x,j)), ndims(x)))
        pad2 = pad(ntuple(j->(j==i ? p[j][2] : size(x,j)), ndims(x)))
        x = cat(pad1, x, pad2; dims=i)
    end
    return x
end

# depthwise convolution: 351ms (x=(56,56,128,128), p=((0,1),(0,1)))
function zeropad3(x, p)
    p = padtuple(p, size(x))
    n = ndims(x)
    d = ntuple(i->(i < n-1 ? 1+sum(p[i]) : i == n-1 ? 1 : size(x,n-1)), n)
    w = fill!(similar(x, d), 0)
    c = ntuple(i->1+p[i][2], n-2)
    w[c...,:,:] .= 1
    conv(w, x; padding=sum.(p), groups=size(x,n-1))
end


# cut-and-paste: med=4.2ms mean=3.8ms (x=(56,56,128,128), p=((0,1),(0,1)))
function zeropad1(x::GPUVal, p)
    p = padtuple(p, size(x))
    n = ndims(x)
    d = ntuple(i->(i < n-1 ? size(x,i)+sum(p[i]) : size(x,i)), n)

    global _y            # assumes p will not change, good for layers
    if !@isdefined(_y) || typeof(_y) !== typeof(x) || size(_y) != d
        _y = similar(x,d)
        fill!(_y, 0)
    end

    y = _y # fill!(similar(x, d), 0)
    c = ntuple(i->(i < n-1 ? (p[i][1]+1:p[i][1]+size(x,i)) : Colon()), n)
    y[c...] .= x
    return y
end

# concatenate: med=18.4ms mean=21.1ms (x=(56,56,128,128), p=((0,1),(0,1)))
function zeropad2(x::GPUVal, p)
    pad(d) = fill!(similar(x, d), 0)
    p = padtuple(p, size(x))

    global _pad                # assume pad/type will not change
    if !@isdefined(_pad) || _pad === nothing
        _pad = Any[Any[nothing,nothing],Any[nothing,nothing]]
    end

    for i in 1:length(p)
        # pad1 = pad(ntuple(j->(j==i ? p[j][1] : size(x,j)), ndims(x)))
        # pad2 = pad(ntuple(j->(j==i ? p[j][2] : size(x,j)), ndims(x)))
        # x = cat(pad1, x, pad2; dims=i)
        if _pad[i][1] === nothing
            _pad[i][1] = pad(ntuple(j->(j==i ? p[j][1] : size(x,j)), ndims(x)))
            _pad[i][2] = pad(ntuple(j->(j==i ? p[j][2] : size(x,j)), ndims(x)))
        end
        x = cat(_pad[i][1], x, _pad[i][2]; dims=i)
    end
    return x
end

# depthwise convolution: med=6.0ms mean=5.4ms (x=(56,56,128,128), p=((0,1),(0,1)))
function zeropad3(x::GPUVal, p)
    p = padtuple(p, size(x))
    n = ndims(x)
    yd = ntuple(i->(i < n-1 ? size(x,i)+sum(p[i]) : size(x,i)), n)

    global _y            # assumes p will not change, good for layers
    if !@isdefined(_y) || typeof(_y) !== typeof(x) || size(_y) != yd
        _y = similar(x,yd)
        fill!(_y, 0)
    end

    global _wpad                # assume padding does not change for testing
    if !@isdefined(_wpad) || _wpad === nothing
        wd = ntuple(i->(i < n-1 ? 1+sum(p[i]) : i == n-1 ? 1 : size(x,n-1)), n)
        _wpad = fill!(similar(x, wd), 0)
        c = ntuple(i->1+p[i][2], n-2)
        _wpad[c...,:,:] .= 1
    end

    conv(_wpad, x; padding=sum.(p), groups=size(x,n-1))
end
