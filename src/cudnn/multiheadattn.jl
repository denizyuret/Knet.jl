import CUDA.CUDNN: cudnnMultiHeadAttnForwardAutoGrad
using  CUDA.CUDNN: cudnnMultiHeadAttnBackwardWeights, cudnnMultiHeadAttnBackwardData, handle, CUDNN_WGRAD_MODE_SET
using AutoGrad: AutoGrad, @primitive1, value


@primitive1((cudnnMultiHeadAttnForwardAutoGrad(weights, queries, keys, values, residuals; dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workSpace, reserveSpace), dout,_out),
            (dweights[] === nothing && cudnnMultiHeadAttnBackward(weights, queries, keys, values, residuals; dout, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workSpace, reserveSpace); dweights[]),
            (dqueries[] === nothing && cudnnMultiHeadAttnBackward(weights, queries, keys, values, residuals; dout, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workSpace, reserveSpace); dqueries[]),
            (dkeys[] === nothing && cudnnMultiHeadAttnBackward(weights, queries, keys, values, residuals; dout, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workSpace, reserveSpace); dkeys[]),
            (dvalues[] === nothing && cudnnMultiHeadAttnBackward(weights, queries, keys, values, residuals; dout, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workSpace, reserveSpace); dvalues[]),
            dout) # dresiduals is equal to dout


# Do all the work in first call
function cudnnMultiHeadAttnBackward(weights, queries, keys, values, residuals; dout, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workSpace, reserveSpace)
    weights, queries, keys, values, residuals, dout = value.((weights, queries, keys, values, residuals, dout))
    # The cudnnMultiHeadAttnBackwardData() function must be invoked after cudnnMultiHeadAttnForward(). The loWinIdx[], hiWinIdx[], queries, keys, values, weights, and reserveSpace arguments should be the same as in the cudnnMultiHeadAttnForward() call. devSeqLengthsDQDO[] and devSeqLengthsDKDV[] device arrays should contain the same start and end attention window indices as devSeqLengthsQO[] and devSeqLengthsKV[] arrays in the forward function invocation.
    dqueries[], dkeys[], dvalues[] = similar(queries), similar(keys), similar(values)
    cudnnMultiHeadAttnBackwardData(handle(), attnDesc, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, oDesc, dout, qDesc, dqueries[], queries, kDesc, dkeys[], keys, vDesc, dvalues[], values, sizeof(weights), weights, sizeof(workSpace), something(workSpace, CU_NULL), sizeof(reserveSpace), something(reserveSpace, CU_NULL))
    # The cudnnMultiHeadAttnBackwardWeights() function should be invoked after cudnnMultiHeadAttnBackwardData(). The queries, keys, values, weights, and reserveSpace arguments should be the same as in cudnnMultiHeadAttnForward() and cudnnMultiHeadAttnBackwardData() calls. The dout argument should be the same as in cudnnMultiHeadAttnBackwardData(). 
    dweights[] = similar(weights)
    addGrad = CUDNN_WGRAD_MODE_SET # TODO: support add mode
    cudnnMultiHeadAttnBackwardWeights(handle(), attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, oDesc, dout, sizeof(weights), weights, dweights[], sizeof(workSpace), something(workSpace, CU_NULL), sizeof(reserveSpace), something(reserveSpace, CU_NULL))
end


# Residuals: The cudnnMultiHeadAttnBackwardData() function does not output partial
# derivatives for residual connections because this result is equal to dout . If the
# multi-head attention model enables residual connections sourced directly from Q, then the
# dout tensor needs to be added to dqueries to obtain the correct result of the latter. This
# operation is demonstrated in the cuDNN multiHeadAttention sample code.
# AutoGrad automatically does the addition.
# What if q = LayerNorm(r)? In my tests dr=dout and dq=0. TODO: check this further.
