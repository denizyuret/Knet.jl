export linear

"""
    linear(w, x; dims=1)

Generalizes matrix multiplication to possibly more than 2 dims with reshapes: `w(A...,B...)
* x(B...,C...) => y(A...,C...)` where `size(w)=(A...,B...)` etc. and `dims=length(B)`. The last
`dims` dimensions of `w` has to match the first `dims` dimensions of `x`.
"""
function linear(w, x; dims=1)
    if ndims(w) > 2 || ndims(x) > 2 || dims > 1
        @assert ndims(w) > dims "ndims(w) must be > dims"
        # ndims(x) may be <= dims in which case assume trailing dims assumed 1 
        ntuple(i->size(x,i), dims) === size(w)[end-dims+1:end] || throw(DimensionMismatch("linear: w=$(size(w)) x=$(size(x)) dims=$dims"))
        w2size = (prod(size(w)[1:end-dims]), prod(size(w)[end-dims+1:end]))
        w2 = (size(w) === w2size ? w : reshape(w, w2size))
        x2 = (size(x,1) === size(w2,2) && ndims(x) <= 2 ? x : reshape(x, size(w2,2), :))
        y2 = w2 * x2
        ysize1 = (dims === ndims(w) ? (1,) : size(w)[1:end-dims])
        ysize2 = (dims >= ndims(x) ? () : size(x)[dims+1:end])
        ysize = (ysize1..., ysize2...)
        y = (size(y2) === ysize ? y2 : reshape(y2, ysize))
    else # handle common matmul case without reshapes
        @assert dims === 1 "dims must be >= 1"
        size(x,1) === size(w,2) || throw(DimensionMismatch("linear: w=$(size(w)) x=$(size(x)) dims=$dims"))
        y = w * x
    end
    return y
end

##
# In RNNs where input has shape (X,B,T) we want the result to be (Y,B,T). In CNNs where
# input has shape (X1,X2,C,B) we want the output to be (Y,B). So in general we want to be
# able to specify how many initial dims of x should be replaced. Does it ever get replaced
# with more than one dim? In my transformer implementation it does.
