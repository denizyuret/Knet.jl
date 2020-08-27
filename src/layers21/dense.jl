export Dense
using Knet.Ops21: mmul
using Knet: atype
using AutoGrad: Param

"""
    Dense(wsize::Integer...; f, dims, dropout, atype, init, binit)
    Dense(weights, bias=nothing; f, dims, dropout)
    (d::Dense)(x)

Generalizes matrix multiplication to possibly more than 2 dims with reshapes: 

    w(M...,N...) * x(N...,K...) => y(M...,K...)

where `wsize=size(weights)=(M...,N...)` and `dims=length(N)`. Optionally dropout is applied
to the input and a bias of size `(M...)` is added and an elementwise activation function `f`
is applied to the output. The last `dims` dimensions of `w` has to match the first `dims`
dimensions of `x` and the size of bias (if there is one) has to match the first
`ndims(w)-dims` dimensions of `w`.

Keyword arguments:
* `f=nothing`: apply activation function to output if not nothing
* `dims=1`: number of input dimensions in the weight tensor
* `dropout=0`: apply dropout with this probability to input if non-zero
* `atype=Knet.atype()`: array and element type for parameter initialization
* `init=𝑼(√(6/(fanin+fanout)))`: initialization function for weights
* `binit=zeros`: initialization function for bias, if `nothing` do not use bias

References:
* torch.nn.Linear
* tf.keras.layers.Dense
"""
struct Dense; w; b; f; dims; dropout; end

function Dense(weights, bias=nothing; f=nothing, dims=1, dropout=0)
    @assert ndims(weights) > dims "ndims(weights) must be > dims"
    @assert bias === nothing || size(bias) === size(weights)[1:end-dims] "weights and bias do not match"
    w = (weights isa Param ? weights : Param(weights))
    b = (bias isa Nothing || bias isa Param ? bias : Param(bias))
    Dense(w, b, f, dims, dropout)
end

function Dense(wsize::Integer...; f=nothing, dims=1, dropout=0, atype=atype(), binit=zeros,
               init=𝑼(√(6/(densein(wsize,dims)+denseout(wsize,dims)))))
    @assert length(wsize) > dims "ndims(weights) must be > dims"
    w = Param(convert(atype,init(wsize...)))
    b = binit isa Nothing ? nothing : Param(convert(atype,binit(wsize[1:end-dims]...)))
    Dense(w, b, f, dims, dropout)
end

function (l::Dense)(x)
    if l.dropout != 0; x = dropout(x, l.dropout); end
    y = mmul(l.w, x)
    if l.b !== nothing; y = y .+ l.b; end
    if l.f !== nothing; y = l.f.(y); end
    return y
end

densein(w,d)=prod(w[end-d+1:end])
denseout(w,d)=prod(w[1:end-d])

# function (l::Dense)(x::MaskedArray)
#     (a,m) = (x.array, x.mask)
#     @assert m===nothing || all(size(m,i) == 1 || size(m,i) == size(a,i) for i in 1:ndims(a))
#     if m === nothing
#         return MaskedArray(l(a), nothing)
#     elseif size(m,1) == 1   # avoid mask multiplication if possible
#         b = l(a)
#         if ndims(b) > ndims(m)
#             m = reshape(m, ntuple(i->1, ndims(b)-ndims(m))..., size(m)...)
#         end
#         return MaskedArray(b, m)
#     else
#         return MaskedArray(l(a .* oftype(a,m)), nothing)
#     end
# end

##
# In RNNs where input has shape (X,B,T) we want the result to be (Y,B,T). In CNNs where
# input has shape (X1,X2,C,B) we want the output to be (Y,B). So in general we want to be
# able to specify how many initial dims of x should be replaced. Does it ever get replaced
# with more than one dim? In my vaswani it does.
