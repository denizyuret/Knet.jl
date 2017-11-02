using Knet
using Knet: @cuda, Cptr, cudnnhandle, cudnnVersion, bytes, TD, FD

function rnn(s, w, x, hx, cx; training=false)  # returns (y, hy, cy, rs)
    (x, xDesc) = rnnInput(x,s)
    (inputSize,batchSize,seqLength) = size(x)
    (hx, hxDesc) = rnnHidden(hx,s,batchSize)
    (cx, cxDesc) = rnnHidden(cx,s,batchSize)
    (w, wDesc) = rnnWeights(w)
    (y, yDesc) = rnnOutput()
    (hy, hyDesc) = rnnHidden(s,batchSize)
    (cy, cyDesc) = rnnHidden(s,batchSize)
    workSpace = rnnWorkSpace()
    if training               # TODO
        reserveSpace = rnnReserveSpace()
        @cuda(cudnn,cudnnRNNForwardTraining,
              (   Cptr,     Cptr,     Cint, Cptr,Cptr,  Cptr,Cptr,  Cptr,Cptr, Cptr,Cptr, Cptr,Cptr,  Cptr,Cptr,  Cptr,Cptr,     Cptr,             Csize_t,        Cptr,                Csize_t),
              s.handle,s.rnnDesc,seqLength,xDesc,   x,hxDesc,  hx,cxDesc,  cx,wDesc,   w,yDesc,   y,hyDesc,  hy,cyDesc,  cy,workspace,workspaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes)
    else
        reserveSpace = nothing
        @cuda(cudnn,cudnnRNNForwardInference,
              (   Cptr,     Cptr,     Cint, Cptr,Cptr,  Cptr,Cptr,  Cptr,Cptr, Cptr,Cptr, Cptr,Cptr,  Cptr,Cptr,  Cptr,Cptr,     Cptr,             Csize_t),
              s.handle,s.rnnDesc,seqLength,xDesc,   x,hxDesc,  hx,cxDesc,  cx,wDesc,   w,yDesc,   y,hyDesc,  hy,cyDesc,  cy,workspace,workspaceSizeInBytes)
    end
    return (y, hy, cy, reserveSpace)
end

rnn_r=recorder(rnn)

rnn(w::Rec, x, hx, cx, s)=(rnn_r(w,x,hx,cx,s; training=true))

rnn(::Type{Grad{1}},dr,r,w,x,hx,cx,s)=((y,hy,cy,rs)=r; (dy,dhy,dcy,drs)=dr; backData(); backWeights(); record in s; return dw)
rnn(::Type{Grad{2}},dr,r,w,x,hx,cx,s)=s.dx
rnn(::Type{Grad{3}},dr,r,w,x,hx,cx,s)=s.dhx
rnn(::Type{Grad{4}},dr,r,w,x,hx,cx,s)=s.dcx


function reshape3d(x)
    n = ndims(x)
    if n==3; x
    elseif n==2; reshape(x, (size(x,1),size(x,2),1))
    elseif n==1; reshape(x, (length(x),1,1))
    else; throw(DimensionMismatch())
    end
end

function rnnInput(x,s)
    x = reshape3d(x)            # input dims: (X,B,T); B (batchsize) and T (seqLength) optional with default=1
    

    xDesc = TD(x)               # TODO: need to create T separate TD(x)'s! Each with cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA)
end    

function rnnHidden(hx,s,batchSize)
    if hx == nothing
        hx = hxDesc = C_NULL
    else
        hx = reshape3d(hx)      # hidden dims: (H,B,L) or (H,B,2L) if bidirectional
        @assert size(hx,1) == s.hiddenSize
        @assert size(hx,2) == batchSize
        @assert ((s.direction == 0 && size(hx,3) == s.numLayers) ||
                 (s.direction == 1 && size(hx,3) == 2*s.numLayers))
        hxDesc = TD(hx)
    end
    return (hx,hxDesc)
end    

function rnnWeights(w)
    wDesc = s.wDesc
    w = s.w                     # w dims: (W,1,1); W given by cudnnGetRNNParamSize
    return (w, wDesc)
end
    

type DD; ptr; states; end       # TODO: Can multiple RNNs share dropout descriptors? Can dropout probability be changed?
function DD(; handle=cudnnhandle(), dropout=0.0, seed=42)
    d = Cptr[0]; s = Csize_t[0]
    @cuda(cudnn,cudnnCreateDropoutDescriptor,(Ptr{Cptr},),d)
    @cuda(cudnn,cudnnDropoutGetStatesSize,(Cptr,Ptr{Csize_t}),handle,s)
    states = KnetArray{UInt8}(s[1]) # TODO: Can this be shared? 638976 bytes.
    @cuda(cudnn,cudnnSetDropoutDescriptor,(Cptr,Cptr,Cfloat,Cptr,Csize_t,Culonglong),
          d[1],handle,dropout,states,bytes(states),seed)
    dd = DD(d[1],states)
    finalizer(dd, x->@cuda(cudnn,cudnnDestroyDropoutDescriptor,(Cptr,),x.ptr))
    return dd
end

type RD; ptr; dropoutDesc; end
function RD(;
            handle=cudnnhandle(),
            hiddenSize=100,
            numLayers=1,
            dropout=0.0,
            inputMode=0,    # CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT = 1    
            direction=0,    # CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL = 1
            mode=0,         # CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3
            algo=0,         # CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
            dataType=0,     # CUDNN_DATA_FLOAT  = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_DATA_HALF   = 2
            seed=42
            )
    dropoutDesc = DD(handle=handle,dropout=dropout,seed=seed)
    d = Cptr[0]
    @cuda(cudnn,cudnnCreateRNNDescriptor,(Ptr{Cptr},),d)
    if cudnnVersion >= 7000
        @cuda(cudnn,cudnnSetRNNDescriptor,(Cptr,Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint,Cint),
              handle,d[1],hiddenSize,numLayers,dropoutDesc.ptr,inputMode,direction,mode,algo,dataType)
    elseif cudnnVersion >= 6000
        @cuda(cudnn,cudnnSetRNNDescriptor_v6,(Cptr,Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint,Cint),
              handle,d[1],hiddenSize,numLayers,dropoutDesc.ptr,inputMode,direction,mode,algo,dataType)
    elseif cudnnVersion >= 5000
        @cuda(cudnn,cudnnSetRNNDescriptor,(Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint),
              d[1],hiddenSize,numLayers,dropoutDesc.ptr,inputMode,direction,mode,dataType)
    else
        error("CUDNN $cudnnVersion does not support RNNs")
    end
    rd = RD(d[1],dropoutDesc)
    finalizer(rd, x->@cuda(cudnn,cudnnDestroyRNNDescriptor,(Cptr,),x.ptr))
    return rd
end

#= CUDNN Interface:

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor(cudnnHandle_t handle,
cudnnRNNDescriptor_t     rnnDesc,      # Input/output
const int                hiddenSize,   # For a single layer
const int                numLayers,    # Number of stacked layers
cudnnDropoutDescriptor_t dropoutDesc,  # No dropout for single layer
cudnnRNNInputMode_t      inputMode,    # CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT = 1    
cudnnDirectionMode_t     direction,    # CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL = 1
cudnnRNNMode_t           mode,         # CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3
cudnnRNNAlgo_t           algo,         # CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
cudnnDataType_t          dataType);    # CUDNN_DATA_FLOAT  = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_DATA_HALF   = 2

# We'll need to support dropout option
#  (does it stop for inference?)
#  How does dropout work?
# How are the weights packed: matrices W,R biases bW,bR
# Why are there two biases bW and bR?
# In BIDIRECTIONAL the forw and back networks share weights?
# CUDNN_LINEAR_INPUT: matrix multiplication at input (where are the weights?)
# CUDNN_SKIP_INPUT: no operation at input.  leading dimension of the input tensor must be equal to the hidden state size.

# RELU: ht = ReLU(Wixt + Riht-1 + bWi + bRi)
# TANH: ht = tanh(Wixt + Riht-1 + bWi + bRi)
# LSTM:
# it = σ(Wixt + Riht-1 + bWi + bRi)
# ft = σ(Wfxt + Rfht-1 + bWf + bRf)
# ot = σ(Woxt + Roht-1 + bWo + bRo)
# c't = tanh(Wcxt + Rcht-1 + bWc + bRc)
# ct = ft◦ct-1 + it◦c't
# ht = ot◦tanh(ct)
# GRU:
# it = σ(Wixt + Riht-1 + bWi + bRu)
# rt = σ(Wrxt + Rrht-1 + bWr + bRr)
# h't = tanh(Whxt + rt◦(Rhht-1 + bRh) + bWh)
# ht = (1 - it)◦h't + it◦ht-1

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInference( cudnnHandle_t handle,

# Inputs x, hx, cx. Outputs y, hy, cy.
# Remember in CNNs CUDNN dims NCHW, Julia dims WHCN.
# In RNNs CUDNN uses NC1, Julia CN.
# Iterations are packed last, CUDNN dims TNC1, Julia CNT.
# Weights are packed in a single W11 tensor, see RNN_example.cu.
# ReserveSpace is being overwritten in training, that's risky for AutoGrad.

const cudnnRNNDescriptor_t rnnDesc,
const int seqLength,                    # Number of iterations to unroll
const cudnnTensorDescriptor_t * xDesc,  # One descriptor per iteration. Dim1: batch, Dim2:veclen
const void * x,                         # Packed data for iterations
const cudnnTensorDescriptor_t hxDesc,   # Initial hidden state of RNN
const void * hx,                        # If hx=NULL initial hidden state = 0
const cudnnTensorDescriptor_t cxDesc,   # Initial cell state for LSTMs
const void * cx,                        # If cx=NULL initial cell state = 0
const cudnnFilterDescriptor_t wDesc,    # Weights
const void * w,                         # Weights
const cudnnTensorDescriptor_t *yDesc,   # One descriptor per iteration for output.
void * y,                               # Packed space for outputs.
const cudnnTensorDescriptor_t hyDesc,   # Final hidden state of RNN
void * hy,                              # If hy=NULL final state not saved.
const cudnnTensorDescriptor_t cyDesc,   # Final cell state for LSTM
void * cy,                              # If cy=NULL final cell state not saved.
void * workspace,                       # Find out workspace size using ...
size_t workSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTraining( cudnnHandle_t handle,
const cudnnRNNDescriptor_t rnnDesc,
const int seqLength,
const cudnnTensorDescriptor_t *xDesc,
const void * x,
const cudnnTensorDescriptor_t hxDesc,
const void * hx,
const cudnnTensorDescriptor_t cxDesc,
const void * cx,
const cudnnFilterDescriptor_t wDesc,
const void * w,
const cudnnTensorDescriptor_t *yDesc,
void * y,
const cudnnTensorDescriptor_t hyDesc,
void * hy,
const cudnnTensorDescriptor_t cyDesc,
void * cy,
void * workspace,
size_t workSpaceSizeInBytes,
void * reserveSpace,
size_t reserveSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData( cudnnHandle_t handle,
const cudnnRNNDescriptor_t rnnDesc,
const int seqLength,
const cudnnTensorDescriptor_t * yDesc,
const void * y,
const cudnnTensorDescriptor_t * dyDesc,
const void * dy,
const cudnnTensorDescriptor_t dhyDesc,
const void * dhy,
const cudnnTensorDescriptor_t dcyDesc,
const void * dcy,
const cudnnFilterDescriptor_t wDesc,
const void * w,
const cudnnTensorDescriptor_t hxDesc,
const void * hx,
const cudnnTensorDescriptor_t cxDesc,
const void * cx,
const cudnnTensorDescriptor_t * dxDesc,
void * dx,
const cudnnTensorDescriptor_t dhxDesc,
void * dhx,
const cudnnTensorDescriptor_t dcxDesc,
void * dcx,
void * workspace,
size_t workSpaceSizeInBytes,
void * reserveSpace,
size_t reserveSpaceSizeInBytes );


cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights( cudnnHandle_t handle,
const cudnnRNNDescriptor_t rnnDesc,
const int seqLength,
const cudnnTensorDescriptor_t * xDesc,
const void * x,
const cudnnTensorDescriptor_t hxDesc,
const void * hx,
const cudnnTensorDescriptor_t * yDesc, 
const void * y,
const void * workspace, 
size_t workSpaceSizeInBytes, 
const cudnnFilterDescriptor_t dwDesc, 
void * dw,
const void * reserveSpace, 
size_t reserveSpaceSizeInBytes );

=#
