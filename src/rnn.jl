### TODO: get rid of these when integrated ################
using Knet

Knet.cudnnhandle()
using Knet: @cuda, cudnnhandle, Cptr, cudnnVersion, bytes, FD, DT, TD, cudnnWorkSpace
using AutoGrad: Rec, Grad, recorder
import Knet.DT
### TODO: get rid of these when integrated ##################

### Size chart (Julia sizes for CUDNN calls)
#
# x: (1,X,B,T) where X = inputSize, B = miniBatch, T = seqLength
# xDesc: Array of T (1,X,B) descriptors
# y: (1,Y,B,T) where Y = hiddenSize * (bidirectional ? 2 : 1)
# yDesc: Array of T (1,Y,B) descriptors
# w: (1,1,W) where W = cudnnGetRNNParamsSize()
# hx,cx,hy,cy: (H,B,L) where H = hidden size, L = numLayers * (bidirectional ? 2 : 1)
#
# Note: cudnn docs say min tensor dims 4 but RNN_example.cu uses 3D tensors
# For Julia calls, x and y do not need the initial 1 dimension and B,T are optional.

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
type RD; ptr::Cptr; end

Base.unsafe_convert(::Type{Cptr}, rd::RD)=rd.ptr

function RD()
    d = Cptr[0]
    @cuda(cudnn,cudnnCreateRNNDescriptor,(Ptr{Cptr},),d)
    rd = RD(d[1])
    finalizer(rd, x->@cuda(cudnn,cudnnDestroyRNNDescriptor,(Cptr,),x.ptr))
    return rd
end


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
          # handle, rnndesc, xdesc, result, dataType
          (Cptr,  Cptr, Cptr, Ptr{Csize_t}, UInt32),
          handle, rd, xd[1], res, dt)
    @cuda(cudnn,cudnnDestroyTensorDescriptor,(Cptr,),xd[1])
    div(res[1], ds)
end

"Keeps an array of 3D tensor descriptors"
type TDs; pvec::Vector{Cptr}; end

Base.unsafe_convert(::Type{Ptr{Cptr}}, tds::TDs)=pointer(tds.pvec)
Base.length(tds::TDs)=length(tds.pvec)

function TDs(a::KnetArray)         # Treat a: (X,B,T) as a 4D array: (1,X,B,T)
    a = reshape(a, 1, size(a,1), size(a,2), size(a,3))
    pvec = Vector{Cptr}(size(a,4))
    @cuda(cudnn,cudnnCreateTensorDescriptor,(Ptr{Cptr},),pvec)
    sz = [Cint(size(a,i)) for i=3:-1:1]
    st = [Cint(stride(a,i)) for i=3:-1:1]
    @cuda(cudnn,cudnnSetTensorNdDescriptor,
          (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint}),
          pvec[1], DT(a), 3, sz, st)
    for i=2:length(pvec); pvec[i]=pvec[1]; end # All descriptors are the same
    tds = TDs(pvec)
    finalizer(tds, x->@cuda(cudnn,cudnnDestroyTensorDescriptor,(Cptr,),x.pvec[1]))
    return tds
end

function TD3(a::KnetArray) # Treat a as a 3D array, pad from right
    n = ndims(a)
    if n==3; TD(a)
    elseif n==2; TD(reshape(a, size(a,1), size(a,2), 1))
    elseif n==1; TD(reshape(a, size(a,1), 1, 1))
    else; throw(DimensionMismatch())
    end
end

function FD3(a::KnetArray) # Treat a as a 3D array, pad from left
    n = ndims(a)
    if n==3; FD(a)
    elseif n==2; FD(reshape(a, 1, size(a,1), size(a,2)))
    elseif n==1; FD(reshape(a, 1, 1, size(a,1)))
    else; throw(DimensionMismatch())
    end
end

function cudnnGetRNNWorkspaceSize(rd::RD, tds::TDs; handle=cudnnhandle())
    res = Csize_t[1]
    @cuda(cudnn, cudnnGetRNNWorkspaceSize,
          # handle, rnndesc, seqLength, xdesc, res        ,
          (Cptr,  Cptr, Cint, Ptr{Cptr}, Ptr{Csize_t}),
          handle, rd, length(tds), tds, res)
    return Int(res[1])
end

function cudnnGetRNNTrainingReserveSize(rd::RD, tds::TDs; handle=cudnnhandle())
    res = Csize_t[1]
    @cuda(cudnn, cudnnGetRNNTrainingReserveSize,
          # handle, rnndesc, seqLength, xdesc, res        ,
          (Cptr,  Cptr, Cint, Ptr{Cptr}, Ptr{Csize_t}),
          handle, rd, length(tds), tds, res)
    return Int(res[1])
end

# layer=numLayers+i+layerid for bidirectionals
# if inputMode=1, first half of the buffers become nothing (no input transform)
function cudnnGetRNNParams{T}(r::RNN, w::KnetArray{T}, layer::Int; handle=cudnnhandle())
    if r.mode == 2
        nws = 8
    elseif r.mode==3
        nws = 6
    else
        nws = 2
    end
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
    weights, biases = [], []
    wdesc = FD(w)
    # The buffers that are overwritten
    matdesc = Cptr[0]
    matptr = Cptr[0]
    dtype = UInt32[0]
    format = UInt32[0]
    ndims = Cint[0]
    dims = Cint[0 for i = 1:3]
    readdims!() = @cuda(cudnn, cudnnGetFilterNdDescriptor,
                       (Cptr, Cint, Ptr{UInt32}, Ptr{UInt32}, #wd, reqdims, dtype, format
                        Ptr{Cint}, Ptr{Cint}), #ndims, dims
                       matdesc[1], 3, dtype, format, ndims, dims)
    for i = 0:nws-1
        # TODO: move them outside of the loop
        @cuda(cudnn, cudnnCreateFilterDescriptor, (Ptr{Cptr},), matdesc)
        # Read the biases
        @cuda(cudnn, cudnnGetRNNLinLayerMatrixParams,
              (Cptr, Cptr, Cint, #handle,rdesc, layer
               Cptr, Cptr, Ptr{T}, #xDesc, wDesc, w
               Cint, Cptr, Ptr{Cptr}), #lid, lmatdesc, linlayermat
              handle, r.rnnDesc, layer,
              xd[1], wdesc, w,
              i, matdesc[1], matptr)
        readdims!()
        if sum(dims) == 0
            push!(weights, nothing)
        else
            sz = Int64.(dims)
            lt = *(sz...)
            if lt == r.hiddenSize^2
                sz = (r.hiddenSize, r.hiddenSize)
            elseif lt == r.hiddenSize*r.inputSize
                sz = (r.hiddenSize, r.inputSize)
            else
                error("Malformed weight array is read")
            end
            ptr = Knet.KnetPtr(matptr[1], *(sz...), gpu(), w)
            push!(weights, KnetArray{T,2}(ptr, sz))
        end

        # Read the biases
        @cuda(cudnn, cudnnGetRNNLinLayerBiasParams,
              (Cptr, Cptr, Cint, #handle,rdesc, layer
               Cptr, Cptr, Ptr{T}, #xDesc, wDesc, w
               Cint, Cptr, Ptr{Cptr}), #lid, lmatdesc, linlayermat
              handle, r.rnnDesc, layer,
              xd[1], wdesc, w,
              i, matdesc[1], matptr)
        readdims!()
        if sum(dims) == 0
            push!(biases, nothing)
        else
            sz = Int64.(dims)
            ptr = Knet.KnetPtr(matptr[1], *(sz...), gpu(), w)
            push!(biases, KnetArray{T,2}(ptr, (sz[1],1)))
        end
    end
    @cuda(cudnn, cudnnDestroyFilterDescriptor, (Cptr,), matdesc[1])
    @cuda(cudnn,cudnnDestroyTensorDescriptor,(Cptr,),xd[1])
    return weights, biases
end


function rnninit(inputSize, hiddenSize;
                 handle=cudnnhandle(),
                 numLayers=1,
                 dropout=0.0,
                 inputMode=0,    # CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT = 1
                 direction=0,    # CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL = 1
                 mode=2,         # CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3
                 algo=0,         # CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
                 dataType=Float32, # CUDNN_DATA_FLOAT  = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_DATA_HALF   = 2
                 seed=42,
                 winit=xavier,
                 binit=zeros
                 )
    # Need to keep dropoutDesc in RNN so it does not get gc'ed.
    dropoutDesc = DD(handle=handle,dropout=dropout,seed=seed)
    rnnDesc = RD()
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
    # Initialize weights
    for i = 0:(r.numLayers * (1 + r.direction) -1)
        mats, bs = cudnnGetRNNParams(r, w, i; handle=cudnnhandle())
        for (m, b) in zip(mats, bs)
            copy!(m, winit(eltype(m), size(m)...))
            copy!(b, binit(eltype(b), size(b)...))
        end
    end
    return (r,w)
end


function rnn{T}(r::RNN, w::KnetArray{T}, x::KnetArray{T}, hx=nothing, cx=nothing;
                handle=cudnnhandle(), training=false)
    # TODO: add some asserts
    seqLength = size(x,3)       # (X,B,T)

    # Input descriptors
    wDesc = FD3(w)              # (1,1,W)
    xtds = TDs(x)               # (1,X,B) x T
    if hx==nothing; hx=hxDesc=C_NULL; else; hxDesc=TD3(hx); end # (H,B,L/2L)
    if cx==nothing; cx=cxDesc=C_NULL; else; cxDesc=TD3(cx); end

    # Output arrays and descriptors
    ysize = collect(size(x))
    ysize[1] = r.hiddenSize * (r.direction == 1 ? 2 : 1)
    y = similar(x, ysize...)    # (H/2H,B,T)
    ytds = TDs(y)               # (1,H/2H,B) x T
    if hx==C_NULL; hy=hyDesc=C_NULL; else; hy=similar(hx); hyDesc=TD3(hy); end
    if cx==C_NULL; cy=cyDesc=C_NULL; else; cy=similar(cx); cyDesc=TD3(cy); end

    # workSpace and reserveSpace
    wss = cudnnGetRNNWorkspaceSize(r.rnnDesc, xtds; handle=handle)
    ws = cudnnWorkSpace(wss)

    if training
        rss = cudnnGetRNNTrainingReserveSize(r.rnnDesc, xtds; handle=handle)
        rs = KnetArray{UInt8}(rss)
        @cuda(cudnn, cudnnRNNForwardTraining,
              (Cptr, Cptr, Cint,  # handle,rnnDesc,seqLength
               Ptr{Cptr}, Ptr{T}, #x
               Cptr, Ptr{T}, #hx
               Cptr, Ptr{T}, #cx
               Cptr, Ptr{T}, #w
               Ptr{Cptr}, Ptr{T}, #y
               Cptr, Ptr{T}, #hy
               Cptr, Ptr{T}, #cy
               Cptr, Csize_t, #ws
               Cptr ,Csize_t#rs
               ),
              handle, r.rnnDesc, seqLength,
              xtds, x,
              hxDesc, hx,
              cxDesc, cx,
              wDesc, w,
              ytds, y,
              hyDesc, hy,
              cyDesc, cy,
              ws, wss,
              rs, rss)
    else
        rs = nothing
        @cuda(cudnn, cudnnRNNForwardInference,
              (Cptr, Cptr, Cint,  # handle,rnnDesc,seqLength
               Ptr{Cptr}, Ptr{T}, #x
               Cptr, Ptr{T}, #h
               Cptr, Ptr{T}, #c
               Cptr, Ptr{T}, #w
               Ptr{Cptr}, Ptr{T}, #y
               Cptr, Ptr{T}, #hy
               Cptr, Ptr{T}, #cy
               Cptr, Csize_t, #ws
               ),
              handle, r.rnnDesc, seqLength,
              xtds, x,
              hxDesc, hx,
              cxDesc, cx,
              wDesc, w,
              ytds, y,
              hyDesc, hy,
              cyDesc, cy,
              ws, wss)
    end
    if hy == C_NULL; hy = nothing; end
    if cy == C_NULL; cy = nothing; end
    return y, hy, cy, rs
end

function rnnback{T}(r::RNN, w::KnetArray{T}, x::KnetArray{T}, y::KnetArray{T}, dy::KnetArray{T},
                    hx, cx, dhy, dcy, rs; handle=cunnhandle(), o...)
    seqLength = size(x,3)       # (X,B,T)

    # Input descriptors:
    wDesc = FD3(w)              # (1,1,W)
    xtds = TDs(x)               # (X,B,T) -> (1,X,B) x T
    ytds = TDs(y)               # (H/2H,B,T) -> (1,H/2H,B) x T
    dytds = TDs(dy)             # TODO: can we use ytds here?
    if hx == nothing; hx=hxDesc=C_NULL; else; hxDesc=TD3(hx); end
    if cx == nothing; cx=cxDesc=C_NULL; else; cxDesc=TD3(cx); end
    if dhy == nothing; dhy=dhyDesc=C_NULL; else; dhyDesc=TD3(dhy); end
    if dcy == nothing; dcy=dcyDesc=C_NULL; else; dcyDesc=TD3(dcy); end

    # Output arrays and descriptors:
    dx = similar(x)             # (X,B,T)
    dxtds = TDs(dx)             # TODO: can we use xtds here?
    dw = similar(w)
    dwDesc = FD3(dw)
    if hx == C_NULL; dhx=dhxDesc=C_NULL; else; dhx=similar(hx); dhxDesc=TD3(dhx); end
    if cx == C_NULL; dcx=dcxDesc=C_NULL; else; dcx=similar(cx); dcxDesc=TD3(dcx); end

    # workSpace and reserveSpace
    ws = cudnnWorkSpace()
    wss = bytes(ws)
    rss = bytes(rs)

    # data backward
    @cuda(cudnn, cudnnRNNBackwardData,
          (Cptr, Cptr, Cint,  # handle, rnnDesc, seqLength
           Ptr{Cptr}, Ptr{T}, #y
           Ptr{Cptr}, Ptr{T}, #dy
           Cptr, Ptr{T}, #dhy
           Cptr, Ptr{T}, #dcy
           Cptr, Ptr{T}, #w
           Cptr, Ptr{T}, #hx
           Cptr, Ptr{T}, #cx
           Ptr{Cptr}, Ptr{T}, #dx
           Cptr, Ptr{T}, #dhx
           Cptr, Ptr{T}, #dcx
           Cptr, Csize_t, #ws
           Cptr, Csize_t), #rs
          # Use rtd with nullables
          handle, r.rnnDesc, seqLength,
          ytds, y,
          dytds, dy,
          dhyDesc, dhy,
          dcyDesc, dcy,
          wDesc, w,
          hxDesc, hx,
          cxDesc, cx,
          dxtds, dx,
          dhxDesc, dhx,
          dcxDesc, dcx,
          ws, wss,
          rs, rss)
    # weights backward
    @cuda(cudnn, cudnnRNNBackwardWeights,
          (Cptr, Cptr, Cint,  # handle, rnnDesc, seqLength
           Ptr{Cptr}, Ptr{T}, #x
           Cptr, Ptr{T}, #hx
           Ptr{Cptr}, Ptr{T}, #y
           Cptr, Csize_t, #ws
           Cptr, Ptr{T}, #dw
           Ptr{Cptr}, Csize_t), #rs
          handle, r.rnnDesc, seqLength,
          xtds, x,
          hxDesc, hx,
          ytds, y,
          ws, wss,
          dwDesc, dw,
          rs, rss)
    # Update the cache
    if dhx==C_NULL; dhx=nothing; end
    if dcx==C_NULL; dcx=nothing; end
    r.dx, r.dhx, r.dcx = dx, dhx, dcx
    return dw
end

function rnn(::Type{Grad{2}}, dt, t, r, w, x, hx=nothing, cx=nothing; o...)
    y,hy,cy,rs = getval(t)
    dy,dhy,dcy,drs = getval(dt)
    w=getval(w); x=getval(x); hx=getval(hx); cx=getval(cx)
    rnnback(r, w, x, y, dy, hx, cx, dhy, dcy, rs; o...)
end

rnn(::Type{Grad{3}}, dt, t, r, w...; o...)=r.dx
rnn(::Type{Grad{4}}, dt, t, r, w...; o...)=r.dhx
rnn(::Type{Grad{5}}, dt, t, r, w...; o...)=r.dcx

rnn_r = recorder(rnn)
rnn(r::RNN, w::Rec, x...; handle=cudnnhandle(), o...)=rnn_r(r, w, x...; handle=handle, training=true)


#=


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
=#



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



=#
