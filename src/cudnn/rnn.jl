import CUDA.CUDNN: cudnnRNNForwardAutoGrad
import AutoGrad: forw
using CUDA.CUDNN: cudnnRNNBackwardWeights_v8, cudnnRNNBackwardData_v8, handle, cudnnRNNTempSpaceSizes, cudnnTempSpace, CUDNN_FWD_MODE_TRAINING, CUDNN_WGRAD_MODE_ADD
using AutoGrad: AutoGrad, @primitive1, value, forwargs, recording, Result, @zerograd


# Define gradients for the AutoGrad method

@primitive1((cudnnRNNForwardAutoGrad(w, x, hx, cx; rnnDesc, fwdMode, devSeqLengths, xDesc, yDesc, y, hDesc, hy, cDesc, cy, workSpace, reserveSpace, dw, dx, dhx, dcx),
             dout,out),
            (cudnnRNNBackward(dout, out, w, x, hx, cx; rnnDesc, fwdMode, devSeqLengths, xDesc, yDesc, y, hDesc, hy, cDesc, cy, workSpace, reserveSpace, dw, dx, dhx, dcx); dw),
            dx, dhx, dcx)


# We do all the work during the backward pass for the first arg
# The cudnnRNNBackwardData_v8() function must be called after cudnnRNNForward().
# The cudnnRNNBackwardWeights_v8() function should be invoked after cudnnRNNBackwardData().

function cudnnRNNBackward(dout, out, w, x, hx, cx; rnnDesc, fwdMode, devSeqLengths, xDesc, yDesc, y, hDesc, hy, cDesc, cy, workSpace, reserveSpace, dw, dx, dhx, dcx)
    (_y,_hy,_cy) = value.(out)
    (dy,dhy,dcy) = value.(dout)
    (w,x,hx,cx) = value.((w,x,hx,cx))
    cudnnRNNBackwardData_v8(handle(), rnnDesc, devSeqLengths, yDesc, _y, dy, xDesc, dx, hDesc, something(hx, CU_NULL), something(dhy, CU_NULL), something(dhx, CU_NULL), cDesc, something(cx, CU_NULL), something(dcy, CU_NULL), something(dcx, CU_NULL), sizeof(w), w, sizeof(workSpace), something(workSpace, CU_NULL), sizeof(reserveSpace), something(reserveSpace, CU_NULL))
    # Currently, the cudnnRNNBackwardWeights_v8() function supports the CUDNN_WGRAD_MODE_ADD mode only so the dweightSpace buffer should be zeroed by the user before invoking the routine for the first time. 
    addGrad = CUDNN_WGRAD_MODE_ADD 
    dw .= 0
    cudnnRNNBackwardWeights_v8(handle(), rnnDesc, addGrad, devSeqLengths, xDesc, x, hDesc, something(hx, CU_NULL), yDesc, _y, sizeof(w), dw, sizeof(workSpace), something(workSpace, CU_NULL), sizeof(reserveSpace), something(reserveSpace, CU_NULL))
    return dw
end

# TODO: can we just use Refs instead of the following?

# cudnnRNNForwardWithDefaults does not have access to AutoGrad state so may not set fwdMode and gradient buffers correctly
# When one of the inputs to cudnnRNNForwardAutoGrad (w, x, hx, cx) is a Value, AutoGrad.forw is called
# If we are computing gradients we need to fix the setup:
# - Set fwdMode to TRAINING
# - Check if workSpace and reserveSpace needs to be resized
# - Alloc gradient buffers if not allocated

function forw(f::typeof(cudnnRNNForwardAutoGrad), w, x, hx, cx; rnnDesc, fwdMode, devSeqLengths, xDesc, yDesc, y, hDesc, hy, cDesc, cy, workSpace, reserveSpace, dw, dx, dhx, dcx)
    args = (w, x, hx, cx)
    (f, nobcast, novalue) = forwargs(f, args)
    @assert nobcast === args              # we shouldn't need to handle broadcasting with rnns
    if recording() && novalue !== nobcast # we are taking gradients and there are tracked args
        fwdMode=CUDNN_FWD_MODE_TRAINING
        (workSpaceSize, reserveSpaceSize) = cudnnRNNTempSpaceSizes(rnnDesc, fwdMode, xDesc)
        if sizeof(reserveSpace) < reserveSpaceSize; reserveSpace = cudnnTempSpace(reserveSpaceSize); end
        if sizeof(workSpace) < workSpaceSize; workSpace = cudnnTempSpace(workSpaceSize); end
        if dw === nothing; dw = similar(w); end
        if dx === nothing; dx = similar(x); end
        if dhx === nothing && hx !== nothing; dhx = similar(hx); end
        if dcx === nothing && cx !== nothing; dcx = similar(cx); end
        v = f(novalue...; rnnDesc, fwdMode, devSeqLengths, xDesc, yDesc, y, hDesc, hy, cDesc, cy, workSpace, reserveSpace, dw, dx, dhx, dcx)
        v = Result(v, f, nobcast, (; rnnDesc, fwdMode, devSeqLengths, xDesc, yDesc, y, hDesc, hy, cDesc, cy, workSpace, reserveSpace, dw, dx, dhx, dcx))
    else
        v = f(novalue...; rnnDesc, fwdMode, devSeqLengths, xDesc, yDesc, y, hDesc, hy, cDesc, cy, workSpace, reserveSpace, dw, dx, dhx, dcx)
    end
    return v
end

import Base: sizeof
@zerograd sizeof(x) # TODO: move this to AutoGrad
