import CUDA.CUDNN: cudnnRNNForwardAutoGrad
using CUDA.CUDNN: cudnnRNNBackwardWeights_v8, cudnnRNNBackwardData_v8, handle, @retry_reclaim
using AutoGrad: AutoGrad, @primitive1, value

# Set fwdMode to TRAINING when one of the inputs w, x, hx, cx is a Value



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
    @retry_reclaim (x->x!=CUDNN_STATUS_SUCCESS) cudnnRNNBackwardData_v8(handle(), rnnDesc, devSeqLengths, yDesc, _y, dy, xDesc, dx, hDesc, something(hx, CU_NULL), something(dhy, CU_NULL), something(dhx, CU_NULL), cDesc, something(cx, CU_NULL), something(dcy, CU_NULL), something(dcx, CU_NULL), sizeof(w), w, sizeof(workSpace), something(workSpace, CU_NULL), sizeof(reserveSpace), something(reserveSpace, CU_NULL))
    # Currently, the cudnnRNNBackwardWeights_v8() function supports the CUDNN_WGRAD_MODE_ADD mode only so the dweightSpace buffer should be zeroed by the user before invoking the routine for the first time. 
    addGrad = CUDNN_WGRAD_MODE_ADD 
    dw .= 0
    @retry_reclaim (x->x!=CUDNN_STATUS_SUCCESS) cudnnRNNBackwardWeights_v8(handle(), rnnDesc, addGrad, devSeqLengths, xDesc, x, hDesc, something(hx, CU_NULL), yDesc, _y, sizeof(w), dw, sizeof(workSpace), something(workSpace, CU_NULL), sizeof(reserveSpace), something(reserveSpace, CU_NULL))
    return dw
end
