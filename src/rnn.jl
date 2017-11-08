### TODO: get rid of these when integrated ################
using Knet

Knet.cudnnhandle()
using Knet: @cuda, cudnnhandle, Cptr, cudnnVersion, bytes, FD, DT, TD
using AutoGrad: Rec, Grad, recorder
import Knet.DT
### TODO: get rid of these when integrated ##################

# Size chart (Julia sizes)
#
# x: (1,X,B,T) where X = inputSize, B = miniBatch, T = seqLength
# xDesc: Array of T (1,X,B) descriptors
# y: (1,Y,B,T) where Y = hiddenSize * (bidirectional ? 2 : 1)
# yDesc: Array of T (1,Y,B) descriptors
# w: (1,1,W) where W = cudnnGetRNNParamsSize()
# hx,cx,hy,cy: (H,B,L) where H = hidden size, L = numLayers * (bidirectional ? 2 : 1)
#
# Note: cudnn docs say min tensor dims 4 but RNN_example.cu uses 3D tensors


"Dropout descriptor"
type DD; ptr::Cptr; states::KnetArray{UInt8,1}; end

Base.unsafe_convert(::Type{Cptr}, dd::DD)=dd.ptr

function DD(; handle=cudnnhandle(), dropout=0.0, seed=42, o...)
    d = Cptr[0]; s = Csize_t[0] # TODO: Can multiple RNNs share dropout descriptors? Can dropout probability be changed?
    @cuda(cudnn,cudnnCreateDropoutDescriptor,(Ptr{Cptr},),d)
    @cuda(cudnn,cudnnDropoutGetStatesSize,(Cptr,Ptr{Csize_t}),handle,s)
    states = KnetArray{UInt8}(s[1]) # TODO: Can this be shared? 638976 bytes.
    @cuda(cudnn,cudnnSetDropoutDescriptor,(Cptr,Cptr,Cfloat,Cptr,Csize_t,Culonglong),
          d[1],handle,dropout,states,bytes(states),seed)
    dd = DD(d[1],states)
    finalizer(dd, x->@cuda(cudnn,cudnnDestroyDropoutDescriptor,(Cptr,),x.ptr))
    return dd
end


"RNN descriptor"
type RD; ptr::Cptr;
    function RD(ptr::Cptr)
        rd = new(ptr)
        finalizer(rd, x->@cuda(cudnn,cudnnDestroyRNNDescriptor,(Cptr,),x.ptr))
        return rd
    end
end
Base.unsafe_convert(::Type{Cptr}, rd::RD)=rd.ptr


"RNN config"
type RNN
    inputSize::Cint
    hiddenSize::Cint
    numLayers::Cint
    dropout::Float64
    inputMode::Cint
    direction::Cint
    mode::Cint
    algo::Cint
    dataType::DataType
    rnnDesc::RD
    dropoutDesc::DD
    dx
    dhx
    dcx
end

function initrnn(inputSize, hiddenSize;
                 handle=cudnnhandle(),
                 numLayers=1,
                 dropout=0.0,
                 inputMode=0,    # CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT = 1    
                 direction=0,    # CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL = 1
                 mode=0,         # CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3
                 algo=0,         # CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
                 dataType=Float32, # CUDNN_DATA_FLOAT  = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_DATA_HALF   = 2
                 seed=42,
                 o...
                 )
    # Need to keep dropoutDesc in RNN so it does not get gc'ed.
    dropoutDesc = DD(handle=handle,dropout=dropout,seed=seed)
    d = Cptr[0]; @cuda(cudnn,cudnnCreateRNNDescriptor,(Ptr{Cptr},),d)
    rnnDesc = RD(d[1])
    if cudnnVersion >= 7000
        @cuda(cudnn,cudnnSetRNNDescriptor,(Cptr,Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint,Cint),
              handle,rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,DT(dataType))
    elseif cudnnVersion >= 6000
        @cuda(cudnn,cudnnSetRNNDescriptor_v6,(Cptr,Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint,Cint),
              handle,rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,DT(dataType))
    elseif cudnnVersion >= 5000
        @cuda(cudnn,cudnnSetRNNDescriptor,(Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint),
              rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,DT(dataType))
    else
        error("CUDNN $cudnnVersion does not support RNNs")
    end
    r = RNN(inputSize,hiddenSize,numLayers,dropout,inputMode,direction,mode,algo,dataType,rnnDesc,dropoutDesc,nothing,nothing,nothing)
    w = KnetArray{dataType}(1,1,cudnnGetRNNParamsLength(r))
    # TODO: initialize weights
    return (w,r)
end

DT(::Type{Float32}) = Cint(0)
DT(::Type{Float64}) = Cint(1)
DT(::Type{Float16}) = Cint(2)

function cudnnGetRNNParamsLength(r::RNN; handle=cudnnhandle())
    res = Csize_t[0]
    xd = Cptr[0]    # xDesc: (1,X,B) where X = inputSize, B is ignored, so assume 1
    xs = r.inputSize
    dt = DT(r.dataType)
    ds = sizeof(r.dataType)
    rd = r.rnnDesc
    @cuda(cudnn,cudnnCreateTensorDescriptor,(Ptr{Cptr},),xd)
    @cuda(cudnn,cudnnSetTensorNdDescriptor,
          # td, dataType, nbDims, dimA, strideA
          (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint}),
          xd[1], dt, 3, Cint[1,xs,1], Cint[xs,1,1])
    @cuda(cudnn, cudnnGetRNNParamsSize,
          # handle, rnndesc, seqlength, xdesc, res
          (Cptr,  Cptr, Cptr, Ptr{Csize_t}, UInt32),
          handle, rd, xd[1], res, dt)
    @cuda(cudnn,cudnnDestroyTensorDescriptor,(Cptr,),xd[1])
    div(res[1], ds)
end

"Keeps an array of 3D tensor descriptors"
immutable TDs; pvec::Vector{Cptr}; end

Base.unsafe_convert(::Type{Cptr}, tds::TDs)=pointer(tds.pvec)
Base.length(tds::TDs)=length(tds.pvec)

function TDs(a::KnetArray)
    n = ndims(a); @assert n==4  # (1,X,B,T)
    d = Vector{Cptr}(size(a,n))
    @cuda(cudnn,cudnnCreateTensorDescriptor,(Ptr{Cptr},),d)
    sz = [Cint(size(a,n-i)) for i=1:n-1]
    st = [Cint(stride(a,n-i)) for i=1:n-1]
    @cuda(cudnn,cudnnSetTensorNdDescriptor,
          (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint}),
          d[1], DT(a), n-1, sz, st)
    for i=2:length(d); d[i]=d[1]; end
    tds = TDs(d)
    finalizer(tds, x->@cuda(cudnn,cudnnDestroyTensorDescriptor,(Cptr,),x.pvec[1]))
    return tds
end

function cudnnGetRNNWorkspaceSize(rd::RD, tds::TDs; handle=cudnnhandle())
    res = Csize_t[1]
    @cuda(cudnn, cudnnGetRNNWorkspaceSize,
          # handle, rnndesc, seqlength, xdesc, res        ,
          (Cptr,  Cptr, Cint, Ptr{Cptr}, Ptr{Csize_t}),
          handle, rd, length(tds), tds, res)
    return Int(res[1])
end

function cudnnGetRNNTrainingReserveSize(rd::RD, tds::TDs; handle=cudnnhandle())
    res = Csize_t[1]
    @cuda(cudnn, cudnnGetRNNTrainingReserveSize,
          # handle, rnndesc, seqlength, xdesc, res        ,
          (Cptr,  Cptr, Cint, Ptr{Cptr}, Ptr{Csize_t}),
          handle, rd, length(tds), tds, res)
    return Int(res[1])
end

# This datatype should only
# contain the read only buffers
# user shouldn't call constructor of this fn
# but rather a high-level function like init_rnn
# should create this
type RNNCache
    rd # rnn descriptor for gpu
    inputSize
    dx; dhx; dcx
end

_xtdims(input_size::Int) = (1,input_size, 1)
# the assumption is parameter size should not depend on sequence
# or batch dimensions
function nparams(rc::RNNCache;
                 xtdims=nothing, #for debugging
                 handle=cudnnhandle(),o...)
    if xtdims==nothing; xtdims = _xtdims(rc.inputSize); end
    eltype = JDT(rc.rd.dataType)
    res = Csize_t[1]
    #tds = Cptr[TD(xtdims).ptr]
    @cuda(cudnn, cudnnGetRNNParamsSize,
          # handle, rnndesc, seqlength, xdesc, res
          (Cptr,  Cptr, Cptr, Ptr{Csize_t}, UInt32),
          handle, rc.rd, RTD(xtdims, rc.rd.dataType), res, rc.rd.dataType)
    return div(Int(res[1]), sizeof(eltype))
end

function init_params(rc::RNNCache; handle=cudnnhandle(),o...)
    eltype = JDT(rc.rd.dataType)
    params = KnetArray{eltype}(1,1,nparams(rc; handle=handle,o...))
    return params
end

# Parameter collection and re-collection stuff
# INCOMPLETE
function get_params{T}(w::KnetArray{T}, rc::RNNCache;
                    handle=cudnnhandle(),
                    o...)
    if rc.rnnType in (0,1)
        nmats = 1
    elseif rc.rnnType == 2
        nmats = 4
    else #gru
        nmats = 3
    end
    if rc.inputMode == 0; nmats = 2nmats; end
    weights = []#Ref{Ptr{Void}}()
    biases = []#Ref{Ptr{Void}}()
    dtype = eltype(params)
    #hweight_size = (rc.hiddenSize, rc.hiddenSize)
    #iweight_size = (rc.inputSize, rc.hiddenSize)
    #sizes = (hweight_size, iweight_size)
    #=for fn in (:cudnnGetRNNLinLayerMatrixParams, :cudnnGetRNNLinLayerBiasParams)
        
        
    end=#
    #=for layer = 1:rc.numLayers
        # existence of a linar layer
        for lid = 1:nmats
                eval(:(
                    widesc = FD() #init w/o set
                    wi = Ref{Ptr{Void}}()
                    widims = Cint[0,0,0]
                    widims2 = Cint[0,0,0]
                    @cuda(cudnn, cudnnGetRNNLinLayerBiasParams,
                          (Cptr, Cptr, Cint,
                           Cptr, Cptr, Ptr{T},
                           Cint, Cptr, Ptr{Ptr{T}}),
                          handle, rc.rd, layer,
                          TD(shape, rc.rd.dataType), FD(w), w,
                          lid, bidesc, bi)
                ))
            end
            # dummy descriptor (will be overwritten by the fn)
            
            
            @cuda(cudnn, cudnnGetRNNLinLayerMatrixParams,cudnnGetFilterNdDescriptor,
                  Cptr, Cint, Ptr{UInt32}, Ptr{UInt32}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}
                  )
            
            # collect weights and biases
            
        end
    end=#
end


function _init_cudnn_rnn(;o...)
    rd = RD(;o...)
    fnames = fieldnames(RNNCache)
    cache = RNNCache([nothing for f in fnames]...)
    cache.rd = rd           
    for (name, value) in o
        if name in fnames
            setfield!(cache, name, value)
        end
    end
    params = init_params(cache; o...)
    return params, cache
end

# initializetion functions
for (mode, fn_name) in zip(Array(0:3),
                           [:init_rnn_relu, :init_rnn_tanh, :init_lstm, :init_gru])
    eval(:(
        ($fn_name)(hidden::Int, input::Int; o...)
            = _init_cudnn_rnn(;o..., mode=($mode), hiddenSize=hidden, inputSize=input)))
end
    

# workspace scope
let
    # TODO: make this shared with cnns?
    workspace = nothing

    # only weight backward will be enoguh due to caching
    global getws, cleanws!, wssize
    
    function getws(wss;o...)
        if workspace == nothing || bytes(workspace) < wss
            workspace = KnetArray{Int8}(wss)
        end
        return workspace
    end

    function cleanws!()
        workspace=nothing
    end

    wssize() = (workspace == nothing) ? 0 : bytes(workspace)
end


function rnn{T}(w::KnetArray{T}, x::KnetArray{T}, hx, cx, cache;
                gets=false,
                training=false,
                handle=cudnnhandle(),
                o...)
    # initialize workspace and reserved space
    seqlength = size(x,3)
    #xtdims = _xtdims(cache.inputSize)#(size(x,1,2)..., 1)
    # TODO: should we share tensor desriptors
    # TODO: padding?
    cT = cache.rd.dataType
    xtdims = (size(x,1), size(x,2),1)
    xtds = [RTD(xtdims, cT) for i=1:seqlength]
    # allocate the workspace
    wss = workspace_size(cache.rd, xtds;o...)
    ws = getws(wss; o...)
    # allocata the reserved spave
    # TODO: can we do better in terms of memory?
    rss = reserved_size(cache.rd, xtds;o...)
    rs = training ? KnetArray{Int8}(rss) : nothing
    # Allocate the output data
    output_size = (cache.rd.hiddenSize*(1+Int(cache.rd.direction)),
                   xtdims[2], seqlength)
    hidden_size = (cache.rd.hiddenSize, xtdims[2], cache.rd.numLayers
                   * (1+Int(cache.rd.direction)))
    if hx == nothing; hx=C_NULL; end
    if cx == nothing; cx=C_NULL; end
    #T = eltype(x)
    #x = KnetArray{T}(hidden_size)
    y = KnetArray{T}(output_size)
    ytds = [RTD(hidden_size, cT) for i=1:seqlength]
    if gets
        hy = KnetArray{T}(hidden_size)
        # only lstms
        cy = (cache.rd.mode==2) ? KnetArray{T}(hidden_size) : C_NULL
    else
        hy = C_NULL
        cy = C_NULL
    end
    if training
        @cuda(cudnn, cudnnRNNForwardTraining,
              (Cptr, Cptr, Cint,
               Ptr{Cptr}, Ptr{T}, #x
               Cptr, Ptr{T}, #h
               Cptr, Ptr{T}, #c
               Cptr, Ptr{T}, #w
               Ptr{Cptr}, Ptr{T}, #y
               Cptr, Ptr{T}, #hy
               Cptr, Ptr{T}, #cy
               Cptr, Csize_t, #ws
               Cptr ,Csize_t#rs
               ),
              handle, cache.rd, seqlength,
              xtds, x,
              RTD(hidden_size, cT), hx,
              RTD(hidden_size, cT), cx,
              FD(w), w,
              ytds, y,
              RTD(hidden_size, cT), hy,
              RTD(hidden_size, cT), cy,
              ws, bytes(ws),
              rs, bytes(rs))
    else
        @cuda(cudnn, cudnnRNNForwardInference,
              (Cptr, Cptr, Cint,
               Ptr{Cptr}, Ptr{T}, #x
               Cptr, Ptr{T}, #h
               Cptr, Ptr{T}, #c
               Cptr, Ptr{T}, #w
               Ptr{Cptr}, Ptr{T}, #y
               Cptr, Ptr{T}, #hy
               Cptr, Ptr{T}, #cy
               Cptr, Csize_t, #ws
               ),
              handle, cache.rd, seqlength,
              xtds, x,
              RTD(hidden_size, cT), hx,
              RTD(hidden_size, cT), cx,
              FD(w), w,
              ytds, y,
              RTD(hidden_size, cT), hy,
              RTD(hidden_size, cT), cy,
              ws, bytes(ws))
    end
    if hy == C_NULL; hy = nothing; end
    if cy == C_NULL; cy = nothing; end
    return y, hy, cy, rs
end

    
function rnn_backw{T}(dy::KnetArray{T}, dhy, dcy, y, w, x, hx, cx, hy, cy, cache, rs;
                      handle=cudnnhandle(),
                      o...)
    #Allocate the necessary buffers
    if dhy == nothing; dhy=C_NULL; end
    if dcy == nothing; dcy=C_NULL; end
    
    seqlength = size(x,3)
    #T = eltype(dy)
    cT = cache.rd.dataType
    # The derivative output buffers
    dx = KnetArray{T}(size(x))
    dw = KnetArray{T}(size(w))
    hidden_size = (output_size[1:2]..., 1)
    dhx, dcx = C_NULL, C_NULL
    if hx !== nothing
        dhx = KnetArray{T}(size(hx))
    end
    if cx !== nothing
        dcx == KnetArray{T}(size(cx))
    end
    xtdims = (size(x,1,2)..., 1)
    xtds = [RTD(xtdims, cT).ptr for i=1:seqlength]
    ytds = [RTD(hidden_size, cT).ptr for i=1:seqlength]
    dytds = [RTD(hidden_size, cT).ptr for i=1:seqlength]
    ws = getws(cache.rd, xtds; o...)
    # data backward
    @cuda(cudnn, cudnnRNNBackwardData,
          (Cptr, Cptr, Cint,
           Ptr{Cptr}, Ptr{T}, #y
           Ptr{Cptr}, Ptr{T}, #dy
           Cptr, Ptr{T}, #dhy
           Cptr, Ptr{T}, #dcy
           Cptr, Ptr{T}, #w
           Cptr, Ptr{T}, #hx
           Cptr, Ptr{T}, #cx
           Cptr, Ptr{T}, #dx
           Cptr, Ptr{T}, #dhx
           Cptr, Ptr{T}, #dcx
           Cptr, Csize_t, #ws
           Cptr, Csize_t), #rs
          # Use rtd with nullables
          handle, cache.rd, seqlength,
          ytds, y,
          dytds, dy,
          TD(dhy), dhy,
          TD(dhy), dcy,
          FD(w), w,
          RTD(hidden_size, cT), hx,
          RTD(hidden_size, cT), cx,
          TD(dx), dx,
          RTD(hidden_size, cT), dhx,
          RTD(size(dcx), cT), dcx,
          ws, bytes(ws),
          rs, bytes(rs))
    # weights backward
    @cuda(cudnn, cudnnRNNBackwardWeights,
          (Cptr, Cptr, Cint,
           Ptr{Cptr}, Ptr{T}, #x
           Cptr, Ptr{T}, #hx
           Ptr{Cptr}, Ptr{T}, #y
           Cptr, Csize_t, #ws
           Cptr, Ptr{T},#w
           Ptr{Cptr}, Csize_t),
          handle, cache.rd, seqlength,
          xtds, x,
          RTD(hidden_size, cT), hx,
          ytds, y,
          ws, bytes(ws),
          FD(dw), dw,
          rs, bytes(rs))
    # Update the cache
    cache.dx, cache.dhx, cache.dcx = dx, dhx, dcx
    return dw
end



let rnn_r = recorder(rnn)
    rnn(w::Rec, x, hx, cx, s;o...) = rnn_r(w, x, hx, cx, s; training=true, o...)
    # The main rnn backward
    function rnn(::Type{Grad{1}},
                 dr, r,  w, x,
                 hx, cx, cache; o...)
        dy, dhy, dcy, drs = dr
        y, hy, cy, rs = r
        return rnn_backw(dy, dhy, dcy, y, w, x,
                         hx, cx, hy, cy, cache, rs;
                         o...)
    end
    rnn(::Type{Grad{2}},dr,r,w,x,hx,cx,cache) = cache.dx
    rnn(::Type{Grad{3}},dr,r,w,x,hx,cx,cache) = cache.dhx
    rnn(::Type{Grad{3}},dr,r,w,x,hx,cx,cache) = cache.dcx
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

#=
JDT(cudnndtype) = (Float32, Float64, Float16)[cudnndtype+1]

type RTD
    ptr
    dims
    dtype
end

function RTD(dims, dtype;
            fpacked=true,
            o...)
    #=if length(dims) == 3 && fpacked
        #    info("dims is 3")
        # fix of the batch dimension
        # for 3d tensors in rnns
        dims = (dims[2], dims[1], dims[3])
    end=#
    d = Cptr[0]
    @cuda(cudnn,cudnnCreateTensorDescriptor,(Ptr{Cptr},),d)
    sz = [Cint(dims[i]) for i=1:length(dims)]
    st = Cint[]
    stride = 1
    for i = 1:length(dims)
        push!(st, Cint(stride))
        stride *= dims[i] 
    end
    reverse!(sz)
    reverse!(st)
    @cuda(cudnn,cudnnSetTensorNdDescriptor,
          (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint}),
          d[1], dtype, length(dims), sz, st)
    td = RTD(d[1], dims, dtype)
    finalizer(td, x->@cuda(cudnn,cudnnDestroyTensorDescriptor,(Cptr,),x.ptr))
    return td
end

unsafe_convert(::Type{Cptr}, td::RTD)=td.ptr
=#

