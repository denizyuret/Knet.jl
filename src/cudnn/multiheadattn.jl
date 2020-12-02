using Knet.KnetArrays: DevArray
# @primitive1 is too inefficient for functions with many args, we will define gradients manually:
using AutoGrad: forw, Value, Arg, recording
import AutoGrad: back
AutoGrad.@zerograd sizeof(x)  # TODO: remove after fixing in AutoGrad


# We do all the work during the backward pass for the first arg
function back(
    ::typeof(cudnnMultiHeadAttnForwardAutoGrad), ::Type{Arg{1}}, dout, _out, 
    weights, queries, keys, values, residuals;
    dweights, dqueries, dkeys, dvalues,
    attnDesc, currIdx, loWinIdx, hiWinIdx,
    devSeqLengthsQO, devSeqLengthsKV,
    qDesc, kDesc, vDesc, oDesc,
    out, workSpace, reserveSpace)

    (weights, queries, keys, values, residuals) = value.((weights, queries, keys, values, residuals))

    cudnnMultiHeadAttnBackwardData(
        handle(), attnDesc,
        loWinIdx, hiWinIdx,
        devSeqLengthsQO, devSeqLengthsKV,
        oDesc, dout,
        qDesc, dqueries, queries,
        kDesc, dkeys, keys,
        vDesc, dvalues, values,
        sizeof(weights), cu_null(weights),
        sizeof(workSpace), cu_null(workSpace),
        sizeof(reserveSpace), cu_null(reserveSpace))

     weights !== nothing && cudnnMultiHeadAttnBackwardWeights(
         handle(), attnDesc,
         CUDNN_WGRAD_MODE_SET,
         qDesc, queries,
         kDesc, keys,
         vDesc, values,
         oDesc, dout,
         sizeof(weights), cu_null(weights), cu_null(dweights),
         sizeof(workSpace), cu_null(workSpace),
         sizeof(reserveSpace), cu_null(reserveSpace))

    return dweights
end

# The backward pass for the other args only return already computed gradients
back(::typeof(cudnnMultiHeadAttnForwardAutoGrad), ::Type{Arg{2}}, x...; dqueries, o...) = dqueries
back(::typeof(cudnnMultiHeadAttnForwardAutoGrad), ::Type{Arg{3}}, x...; dkeys, o...) = dkeys
back(::typeof(cudnnMultiHeadAttnForwardAutoGrad), ::Type{Arg{4}}, x...; dvalues, o...) = dvalues
back(::typeof(cudnnMultiHeadAttnForwardAutoGrad), ::Type{Arg{5}}, dout, _out, weights, queries, keys, values, residuals; o...) =
    (residuals === nothing ? nothing : dout)


# Residuals: The cudnnMultiHeadAttnBackwardData() function does not output partial
# derivatives for residual connections because this result is equal to dout . If the
# multi-head attention model enables residual connections sourced directly from Q, then the
# dout tensor needs to be added to dqueries to obtain the correct result of the latter. This
# operation is demonstrated in the cuDNN multiHeadAttention sample code.
# AutoGrad automatically does the addition.
# What if q = LayerNorm(r)? In my tests dr=dout and dq=0. TODO: check this further.
