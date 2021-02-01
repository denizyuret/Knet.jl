export Dense
using Knet.Ops21: mmul
using AutoGrad: Param

"""
    Dense(inputsize, outputsize; winit, binit, activation, dropout)
    Dense(w; bias, inputsize, outputsize, activation, dropout)

Return a function that generalizes matrix multiplication to possibly more than 2 dims with
reshapes:

    w(M...,N...) * x(N...,K...) => y(M...,K...)

where `inputsize=(N...)`, `outputsize=(M...)` each of which can be a single dimension or a
tuple of dimensions. Optionally dropout is applied to the input, a bias of size `(M...)` is
added to the output and an elementwise activation function `activation` is applied to the
output.

The first form takes the sizes and initializes `w` and `bias` using the distributions given
by `winit` and `binit` with an array type that matches the first input. The second form
takes existing weights `w` and `bias` and performs no initialization.

Arguments:
* `w`: the linear transformation weights
* `bias=nothing`: optional bias weights
* `winit=𝑼(√(6/(prod(inputsize)+prod(outputsize))))`: distribution for weight initialization
* `binit=nothing`: distribution for bias initialization, `nothing` means no bias
* `inputsize=size(w)[end]`: size of the input tensor (excluding batch, beam etc. dimensions)
* `outputsize=size(w)[1:end-1]`: size of the output tensor (excluding batch, beam etc. dimensions)
* `activation=nothing`: broadcast activation function to output unless equal to `nothing`
* `dropout=0`: apply dropout with this probability to input if non-zero

References:
* torch.nn.Linear
* tf.keras.layers.Dense
"""
mutable struct Dense
    w
    bias
    winit
    binit
    inputsize
    outputsize
    activation
    dropout
end


function Dense(
    w;
    inputsize=size(w)[end],
    outputsize=size(w)[1:end-1],
    bias=nothing,
    activation=nothing,
    dropout=0,
)
    @assert size(w) == (outputsize..., inputsize...) "size(w) must be $((outputsize..., inputsize...))"
    @assert bias === nothing || bsimilar(size(bias), outputsize) "size(bias) must be $(outputsize) not $(size(bias))"
    w = (w isa Param ? w : Param(w))
    bias = (bias isa Nothing || bias isa Param ? bias : Param(bias))
    Dense(w, bias, nothing, nothing, inputsize, outputsize, activation, dropout)
end


function Dense(
    inputsize, outputsize;
    winit=𝑼(√(6/(prod(inputsize)+prod(outputsize)))),
    binit=nothing,
    activation=nothing,
    dropout=0,
)
    Dense(nothing, nothing, winit, binit, inputsize, outputsize, activation, dropout)
end


function (l::Dense)(x)
    initdense(l, x)
    if l.dropout != 0; x = dropout(x, l.dropout); end
    y = mmul(l.w, x)
    if l.bias !== nothing; y = y .+ l.bias; end
    if l.activation !== nothing; y = l.activation.(y); end
    return y
end


function initdense(l::Dense, x)
    if l.w === nothing
        wsize = (l.outputsize..., l.inputsize...)
        l.w = Param(copyto!(similar(x, wsize...), l.winit(eltype(x), wsize...)))
    end
    if l.bias === nothing && l.binit !== nothing
        bsize = l.outputsize
        l.bias = Param(copyto!(similar(x, bsize...), l.binit(eltype(x), bsize...)))
    end
end


function bsimilar(a,b)          # compare dimensions ignoring trailing ones
    dimpair(d) = (d isa Integer ? (d,()) : d === () ? (1,()) : (d[1],d[2:end]))
    a === () && b === () && return true
    a1,a2 = dimpair(a)
    b1,b2 = dimpair(b)
    a1 == b1 && bsimilar(a2,b2)
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
