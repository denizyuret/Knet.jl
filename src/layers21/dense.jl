export Dense
# TODO: rething param/Param and generally parameter initialization

"""
    Dense(wsize::Integer...; bias=true, activation=nothing, dims=1)
    Dense(weights, bias=nothing; activation=nothing, dims=1)

Generalizes matrix multiplication to possibly more than 2 dims with reshapes: `w(A...,B...)
* x(B...,C...) => y(A...,C...)` where `wsize=(A...,B...)` and `dims=length(B)`. The last
`dims` dimensions of `w` has to match the first `dims` dimensions of `x`. Optionally a bias
of size `(A...)` is added and an elementwise activation function is applied to the
result. The size of bias has to match the first `ndims(w)-dims` dimensions of `w`. If bias
or activation are `nothing`, they are skipped.

References:
* torch.nn.Linear
* tf.keras.layers.Dense
"""
struct Dense; w; b; f; dims; wsize; end

# Store 2-D weights and 1-D bias to avoid reshapes at runtime, keep layer shape in wsize, dims

function Dense(weights, bias=nothing; activation=nothing, dims=1)
    @assert dims <= ndims(weights)
    wsize = size(weights)
    w2 = Param(reshape(weights, prod(wsize[1:end-dims]), prod(wsize[end-dims+1:end])))
    if bias === nothing
        b1 = nothing
    else
        @assert size(bias) === wsize[1:end-dims]
        b1 = Param(vec(bias))
    end
    Dense(w2, b1, activation, dims, wsize)
end

function Dense(wsize::Integer...; bias=true, activation=nothing, dims=1)
    @assert dims <= length(wsize)
    w2 = param(prod(wsize[1:end-dims]), prod(wsize[end-dims+1:end]))
    b1 = bias ? param0(prod(wsize[1:end-dims])) : nothing
    Dense(w2, b1, activation, dims, wsize)
end

# Do not reshape w/x/y unnecessarily, assume trailing dims of x are 1.

function (l::Dense)(x)
    if length(l.wsize) > 2 || ndims(x) > 2 || l.dims > 1
        @assert ntuple(i->size(x,i), l.dims) === l.wsize[end-l.dims+1:end]
        x2 = (size(x,1) === size(l.w,2) && ndims(x) <= 2 ? x : reshape(x, size(l.w,2), :))
        y = l.w * x2
        if l.b !== nothing; y = y .+ l.b; end
        if l.f !== nothing; y = l.f.(y); end
        ysize1 = (l.dims === length(l.wsize) ? (1,) : l.wsize[1:end-l.dims])
        ysize2 = (l.dims >= ndims(x) ? () : size(x)[l.dims+1:end])
        ysize = (ysize1..., ysize2...)
        y = (size(y) === ysize ? y : reshape(y, ysize))
    else # handle common matmul case without reshapes
        @assert size(x,1) === size(l.w,2)
        y = l.w * x
        if l.b !== nothing; y = y .+ l.b; end
        if l.f !== nothing; y = l.f.(y); end
    end
    return y
end

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
