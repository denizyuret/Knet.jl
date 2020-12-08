import CUDA.CUDNN: cudnnMultiHeadAttnForwardAutoGrad
using  CUDA.CUDNN: cudnnMultiHeadAttnBackwardWeights, cudnnMultiHeadAttnBackwardData, cudnnMultiHeadAttnBuffers, handle, CUDNN_WGRAD_MODE_SET
using AutoGrad: AutoGrad, @primitive1, value


@primitive1((cudnnMultiHeadAttnForwardAutoGrad(weights, queries, keys, values, residuals; dready, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workspace, reserveSpace), dout, _out),
            (dready[] || cudnnMultiHeadAttnBackward(weights, queries, keys, values, residuals; dout, dready, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workspace, reserveSpace); dweights[]),
            (dready[] || cudnnMultiHeadAttnBackward(weights, queries, keys, values, residuals; dout, dready, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workspace, reserveSpace); dqueries[]),
            (dready[] || cudnnMultiHeadAttnBackward(weights, queries, keys, values, residuals; dout, dready, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workspace, reserveSpace); dkeys[]),
            (dready[] || cudnnMultiHeadAttnBackward(weights, queries, keys, values, residuals; dout, dready, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workspace, reserveSpace); dvalues[]),
            dout) # dresiduals is equal to dout


# Do all the work in first call
function cudnnMultiHeadAttnBackward(weights, queries, keys, values, residuals; dout, dready, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workspace, reserveSpace)
    # Make sure backward gets called only once
    dready[] && return
    dready[] = true
    # Read all relevant inputs
    (weights, queries, keys, values, residuals, dout) = value.((weights, queries, keys, values, residuals, dout))
    # Allocate gradient buffers if necessary
    if dqueries[] === nothing; dqueries[] = similar(queries); end; @assert issimilar(queries, dqueries[])
    if dkeys[] === nothing; dkeys[] = similar(keys); end; @assert issimilar(keys, dkeys[])
    if dvalues[] === nothing; dvalues[] = similar(values); end; @assert issimilar(values, dvalues[])
    if weights !== nothing && dweights[] === nothing; dweights[] = similar(weights); end; @assert issimilar(weights, dweights[])
    # The cudnnMultiHeadAttnBackwardData() function must be invoked after cudnnMultiHeadAttnForward(). The loWinIdx[], hiWinIdx[], queries, keys, values, weights, and reserveSpace arguments should be the same as in the cudnnMultiHeadAttnForward() call. devSeqLengthsDQDO[] and devSeqLengthsDKDV[] device arrays should contain the same start and end attention window indices as devSeqLengthsQO[] and devSeqLengthsKV[] arrays in the forward function invocation.
    cudnnMultiHeadAttnBackwardData(handle(), attnDesc, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, oDesc, dout, qDesc, dqueries[], queries, kDesc, dkeys[], keys, vDesc, dvalues[], values, sizeof(weights), something(weights, CU_NULL), sizeof(workspace), something(workspace, CU_NULL), sizeof(reserveSpace), something(reserveSpace, CU_NULL))
    # The cudnnMultiHeadAttnBackwardWeights() function should be invoked after cudnnMultiHeadAttnBackwardData(). The queries, keys, values, weights, and reserveSpace arguments should be the same as in cudnnMultiHeadAttnForward() and cudnnMultiHeadAttnBackwardData() calls. The dout argument should be the same as in cudnnMultiHeadAttnBackwardData(). 
    if weights !== nothing
        addGrad = CUDNN_WGRAD_MODE_SET # TODO: support add mode
        cudnnMultiHeadAttnBackwardWeights(handle(), attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, oDesc, dout, sizeof(weights), weights, dweights[], sizeof(workspace), something(workspace, CU_NULL), sizeof(reserveSpace), something(reserveSpace, CU_NULL))
    end
end


# cudnnMultiHeadAttnForwardWithDefaults does not have access to AutoGrad state so may not set training mode correctly
# When one of the main arguments to cudnnMultiHeadAttnForwardAutoGrad is a Value, AutoGrad.forw is called
# If we are computing gradients we need to fix the setup:
# - reserveSpace should be NULL in inference mode and non-NULL in the training mode. 
# - Check if workspace and reserveSpace needs to be resized.
# - Alloc gradient buffers if not allocated (this is done in backward)

function forw(f::typeof(cudnnMultiHeadAttnForwardAutoGrad), weights, queries, keys, values, residuals; dready, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workspace, reserveSpace)
    args = (weights, queries, keys, values, residuals)
    (f, nobcast, novalue) = forwargs(f, args)
    @assert nobcast === args    # we shouldn't need to handle broadcasting
    @assert novalue !== args    # there should be some tracked args for forw to be called
    # Cannot use @workspace here because it is shared between forw and back calls
    (weightSize, workspaceSize, reserveSpaceSize) = cudnnMultiHeadAttnBuffers(attnDesc, training=recording())
    if workspaceSize > 0 && workspace === nothing; workspace = cudnnTempSpace(workspaceSize); end
    if reserveSpaceSize > 0 && reserveSpace === nothing; reserveSpace = cudnnTempSpace(reserveSpaceSize); end
    # The cudnn API decides training/inference based on reserveSpace
    if !recording() && reserveSpace !== nothing; reserveSpace = nothing; end
    if recording() && reserveSpace === nothing; reserveSpace = cudnnTempSpace(16); end
    @assert sizeof(weights) >= weightSize  "weights should be at least $weightSize bytes."
    @assert sizeof(workspace) >= workspaceSize  "worksSpace should be at least $workspaceSize bytes"
    @assert sizeof(reserveSpace) >= reserveSpaceSize  "reserveSpace should be at least $reserveSpaceSize bytes"
    v = f(novalue...; dready, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workspace, reserveSpace)
    if recording()
        v = Result(v, f, nobcast, (; dready, dweights, dqueries, dkeys, dvalues, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, kDesc, vDesc, oDesc, out, workspace, reserveSpace))
    end
    return v
end


# Residuals: The cudnnMultiHeadAttnBackwardData() function does not output partial
# derivatives for residual connections because this result is equal to dout . If the
# multi-head attention model enables residual connections sourced directly from Q, then the
# dout tensor needs to be added to dqueries to obtain the correct result of the latter. This
# operation is demonstrated in the cuDNN multiHeadAttention sample code.
# AutoGrad automatically does the addition.
# What if q = LayerNorm(r)? In my tests dr=dout and dq=0. TODO: check this further.
