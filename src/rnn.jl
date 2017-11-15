# TODO: document exported functions

### Size chart (Julia sizes for CUDNN calls)
# Note: For Julia calls, x and y do not need the initial 1 dimension and B,T are optional.
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

function DD(; handle=cudnnhandle(), dropout=0.0, seed=0, o...)
    if seed==0; seed=floor(Culonglong,time()); end
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

function cudnnGetRNNParamsLength(r::RNN; handle=cudnnhandle())
    res = Csize_t[0]
    xDesc = TD(r.dataType, 1, r.inputSize, 1)    # xDesc: (1,X,B) where X = inputSize, B is ignored, so assume 1
    @cuda(cudnn, cudnnGetRNNParamsSize,
          # handle, rnndesc, xdesc, result, dataType
          (Cptr,  Cptr, Cptr, Ptr{Csize_t}, UInt32),
          handle, r.rnnDesc, xDesc, res, DT(r.dataType))
    div(res[1], sizeof(r.dataType))
end

"Keeps an array of 3D tensor descriptors"
type TDs; pvec::Vector{Cptr}; xDesc::Vector{TD}; end     # Keep xDesc in TDs so it does not get gc'ed

Base.unsafe_convert(::Type{Ptr{Cptr}}, tds::TDs)=pointer(tds.pvec)
Base.length(tds::TDs)=length(tds.pvec)

function TDs{A}(x::KnetArray{A},::Void) # Treat x: (X,B?,T?) as a 4D array: (1,X,B,T)
    xDesc = TD(A,1,size(x,1),size(x,2)) # we can use a single xDesc
    pvec = Vector{Cptr}(size(x,3))
    pvec[:] = xDesc.ptr
    return TDs(pvec, [xDesc])
end

function TDs{A}(x::KnetArray{A},batchSizes::Vector{Int}) # x: (X,B*), batchSizes gives us Bt sizes
    @assert sum(batchSizes) == div(length(x),size(x,1))
    X = size(x,1)
    xs = [ TD(A,1,X,B) for B in batchSizes ]
    ps = [ xd.ptr for xd in xs ]
    return TDs(ps,xs)
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

# Return eltype,size
function cudnnGetFilterNdDescriptor(wDesc::FD; nbDimsRequested = 8)
    dataType = Cint[0]
    format = Cint[0]
    nbDims = Cint[0]
    filterDimA = Vector{Cint}(nbDimsRequested)
    @cuda(cudnn, cudnnGetFilterNdDescriptor,
          (Cptr, Cint, Ptr{UInt32}, Ptr{UInt32}, Ptr{Cint}, Ptr{Cint}),
          wDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)
    if nbDims[1] > nbDimsRequested
        cudnnGetFilterNdDescriptor(wDesc::FD; nbDimsRequested = nbDims[1])
    else
        (Float32,Float64,Float16)[1+dataType[1]],
        (filterDimA[nbDims[1]:-1:1]...)
    end
end

"""

    cudnnGetRNNParam{T}(r::RNN, w::KnetArray{T}, layer, id, matrix=true)

Return a single weight matrix or bias vector as a subarray of w.

Valid `id` values:
* For RELU and TANH RNNs, input = 0, hidden = 1.
* For GRU reset = 0,3; update = 1,4; newmem = 2,5; 0:2 for input, 3:5 for hidden
* For LSTM inputgate = 0,4; forget = 1,5; newmem = 2,6; output = 3,7; 0:3 for input, 4:7 for hidden

Valid `layer` values:
* For direction=0 (uni) RNNs 0:(numLayers-1)
* For direction=1 (bi)  RNNs 0:(2*numLayers-1), forw and back layers alternate.

The effect of inputMode: Let I=0 for RELU/TANH, 0:2 for GRU, 0:3 for LSTM
* For inputMode=0, param(0,I) is a (hiddenSize,inputSize) matrix.
* For inputMode=1, param(0,I) is empty.
* For direction=1 (bi), the same applies to param(1,I): the first back layer.

"""

function cudnnGetRNNParam(r::RNN, w, layer, id, matrix=true; handle=cudnnhandle())
    T = eltype(w) # w could be a Rec so w::KnetArray{T} is not an option
    xDesc = TD(T,1,r.inputSize,1)
    wDesc = FD(T,1,1,length(w))
    paramDesc = FD(T,1,1,1,1)
    param = Cptr[0]
    if matrix
        @cuda(cudnn, cudnnGetRNNLinLayerMatrixParams,
              (Cptr, Cptr, Cint, #handle,rdesc, layer
               Cptr, Cptr, Cptr, #xDesc, wDesc, w
               Cint, Cptr, Ptr{Cptr}), #lid, lmatdesc, linlayermat
              handle, r.rnnDesc, layer,
              xDesc, wDesc, getval(w),
              id, paramDesc, param)
    else
        @cuda(cudnn, cudnnGetRNNLinLayerBiasParams,
              (Cptr, Cptr, Cint, #handle,rdesc, layer
               Cptr, Cptr, Cptr, #xDesc, wDesc, w
               Cint, Cptr, Ptr{Cptr}), #lid, lmatdesc, linlayermat
              handle, r.rnnDesc, layer,
              xDesc, wDesc, getval(w),
              id, paramDesc, param)
    end
    dt,sz = cudnnGetFilterNdDescriptor(paramDesc)
    len = prod(sz)
    i1 = 1 + div(Int(param[1] - pointer(w)), sizeof(T))
    i2 = i1 + len - 1
    if len == 1 # empty weights when inputMode=1 show up as size (1,1,1)
        nothing
    elseif matrix
        h = Int(r.hiddenSize)
        reshape(w[i1:i2], (div(len,h),h)) # weight matrices are transposed?
    else # bias
        w[i1:i2]
    end
end


"""

    cudnnGetRNNParams(r::RNN, w)

Split w into individual parameters and return them as an array.

The order of params returned (subject to change):
* All weight matrices come before all bias vectors.
* Matrices and biases are sorted lexically based on (layer,id).
* See @doc cudnnGetRNNParam for valid layer and id values.
* Input multiplying matrices are `nothing` if r.inputMode = 1.

"""
function cudnnGetRNNParams(r::RNN, w; handle=cudnnhandle())
    layers = r.numLayers * (r.direction == 1 ? 2 : 1)
    ids = r.mode == 2 ? 8 : r.mode == 3 ? 6 : 2
    ws = []
    for m in (true, false)
        for l in 0:layers-1
            for i in 0:ids-1
                push!(ws, cudnnGetRNNParam(r, w, l, i, m; handle=handle))
            end
        end
    end
    return ws
end


"""

    rnninit(inputSize, hiddenSize; opts...)

Return an `(r,w)` pair where `r` is a RNN struct and `w` is a single weight
array that includes all matrices and biases for the RNN.

# Keyword Arguments:
- `rnnType=:lstm` Type of RNN: One of :relu, :tanh, :lstm, :gru.
- `numLayers=1`: Number of RNN layers.
- `bidirectional=false`: Create a bidirectional RNN if `true`.
- `dropout=0.0`: Dropout probability. Ignored if `numLayers==1`.
- `skipInput=false`: Do not multiply the input with a matrix if `true`.
- `dataType=Float32`: Data type to use for weights.
- `algo=0`: Algorithm to use, see CUDNN docs for details.
- `seed=0`: Random number seed. Uses `time()` if 0.
- `winit=xavier`: Weight initialization method for matrices.
- `bias=ones`: Weight initialization method for bias vectors.

"""
function rnninit(inputSize, hiddenSize;
                 handle=cudnnhandle(),
                 numLayers=1,
                 dropout=0.0,
                 skipInput=false,     # CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT = 1
                 bidirectional=false, # CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL = 1
                 rnnType=:lstm,       # CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3
                 dataType=Float32,    # CUDNN_DATA_FLOAT  = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_DATA_HALF   = 2
                 algo=0,              # CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
                 seed=0,              # seed=0 for random init, positive integer for replicability
                 winit=xavier,
                 binit=ones
                 )
    # Need to keep dropoutDesc in RNN so it does not get gc'ed.
    dropoutDesc = DD(handle=handle,dropout=dropout,seed=seed)
    inputMode = skipInput ? 1 : 0
    direction = bidirectional ? 1 : 0
    mode = findfirst((:relu,:tanh,:lstm,:gru), rnnType) - 1
    if mode < 0; error("rnninit: Valid modes are :relu,:tanh,:lstm,:gru"); end
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
    for a in cudnnGetRNNParams(r,w; handle=handle)
        if a == nothing
            continue
        elseif ndims(a) == 2
            copy!(a, winit(dataType, size(a)))
        elseif ndims(a) == 1
            copy!(a, binit(dataType, size(a)))
        else
            error()
        end
    end
    return (r,w)
end


"""

    rnn(r, w, x[, hx, cx]; batchSizes=nothing)

Returns a tuple (y,hy,cy,rs) given rnn `r`, weights `w`, input `x` and
optionally the hidden state `hx` and cell state `cx`.  `r` and `w` should
come from a previous call to `rnninit`. `cx` is only used in LSTMs.  Both
`hx` and `cx` are optional, they are treated as zero arrays if not provided.
`hy` and `cy` will be `nothing` if `hx` and `cx` are not explicitly provided.
`rs` is a buffer the RNN needs for its gradient calculation.  The input and
output dimensions are:

- `x`: (X,[B,T])
- `y`: (H/2H,[B,T])
- `hx`,`cx`,`hy`,`cy`: (H,B,L/2L)
- `batchSizes`: `nothing` or `Vector{Int}(T)`

where W is the length of RNN weights, X is inputSize, H is hiddenSize, B is
batchSize, T is seqLength, L is numLayers.  `x` can be 1, 2, or 3
dimensional.  If `batchSizes==nothing`, a 1-D `x` represents a single
instance, a 2-D `x` represents a single minibatch, and a 3-D `x` represents a
sequence of identically sized minibatches.  If `batchSizes` is an array of
(non-increasing) integers, it gives us the batch size for each time step in a
sequence.  In that case `div(length(x),size(x,1))` should equal
`sum(batchSizes)`. `y` gives us the hidden vector at each time step and has
the same dimensionality as `x`, differing only in its first dimension, which
is H if the rnn is unidirectional, 2H if bidirectional.  Hidden vectors
`hx`,`cx` etc. are size (H,B,L) for unidirectional RNNs, and (H,B,2L) for
bidirectional RNNs.  

"""
function rnn{T}(r::RNN, w::KnetArray{T}, x::KnetArray{T},
                hx::Union{KnetArray{T},Void}=nothing,
                cx::Union{KnetArray{T},Void}=nothing;
                batchSizes=nothing,
                handle=cudnnhandle(), training=false)
    # TODO: add some asserts

    seqLength = batchSizes==nothing ? size(x,3) : length(batchSizes) # (X,B,T) or (X,B+) with batchSizes

    # Input descriptors
    wDesc = FD3(w)              # (1,1,W)
    xtds = TDs(x,batchSizes)    # (1,X,Bt) x T
    if hx==nothing; hx=hxDesc=C_NULL; else; hxDesc=TD3(hx); end # (H,B,L/2L)
    if cx==nothing || r.mode != 2; cx=cxDesc=C_NULL; else; cxDesc=TD3(cx); end

    # Output arrays and descriptors
    ysize = collect(size(x))
    ysize[1] = r.hiddenSize * (r.direction == 1 ? 2 : 1)
    y = similar(x, ysize...)    # (H/2H,B,T)
    ytds = TDs(y,batchSizes)    # (1,H/2H,B) x T
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
                    hx, cx, dhy, dcy, rs; handle=cunnhandle(), batchSizes=nothing, o...)
    seqLength = size(x,3)       # (X,B,T)

    # Input descriptors:
    wDesc = FD3(w)              # (1,1,W)
    xtds = TDs(x,batchSizes)    # (X,B,T) -> (1,X,B) x T
    ytds = TDs(y,batchSizes)    # (H/2H,B,T) -> (1,H/2H,B) x T
    dytds = TDs(dy,batchSizes)  # TODO: can we use ytds here?
    if hx == nothing; hx=hxDesc=C_NULL; else; hxDesc=TD3(hx); end
    if cx == nothing; cx=cxDesc=C_NULL; else; cxDesc=TD3(cx); end
    if dhy == nothing; dhy=dhyDesc=C_NULL; else; dhyDesc=TD3(dhy); end
    if dcy == nothing; dcy=dcyDesc=C_NULL; else; dcyDesc=TD3(dcy); end

    # Output arrays and descriptors:
    dx = similar(x)             # (X,B,T)
    dxtds = TDs(dx,batchSizes)  # TODO: can we use xtds here?
    dw = zeros(w)               # dw is used additively
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

let rnn_r = recorder(rnn); global rnn
    rnn(r::RNN, w...; handle=cudnnhandle(), o...)=rnn_r(r, w...; handle=handle, training=true)
end

# We need x[:,:,t] and hx[:,:,l]
using Knet: Index3
import Base: getindex
function getindex{T}(A::KnetArray{T}, ::Colon, ::Colon, I::Index3)
    B = reshape(A, stride(A,3), size(A,3))
    reshape(B[:,I], size(A,1), size(A,2))
end
