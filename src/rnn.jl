# TODO: 
# finish cpu implementation.
# make RNN objects callable?

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
function cudnnGetRNNParams(r::RNN, w::KnetArray; handle=cudnnhandle())
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
array that includes all matrices and biases for the RNN. Keyword arguments:

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

RNNs compute the output h[t] for a given iteration from the recurrent
input h[t-1] and the previous layer input x[t] given matrices W, R and
biases bW, bR from the following equations:

`:relu` and `:tanh`: Single gate RNN with activation function f:

    h[t] = f(W * x[t] .+ R * h[t-1] .+ bW .+ bR)

`:gru`: Gated recurrent unit:

    i[t] = sigm(Wi * x[t] .+ Ri * h[t-1] .+ bWi .+ bRi) # input gate
    r[t] = sigm(Wr * x[t] .+ Rr * h[t-1] .+ bWr .+ bRr) # reset gate
    n[t] = tanh(Wn * x[t] .+ r[t] .* (Rn * h[t-1] .+ bRn) .+ bWn) # new gate
    h[t] = (1 - i[t]) .* n[t] .+ i[t] .* h[t-1]

`:lstm`: Long short term memory unit with no peephole connections:

    i[t] = sigm(Wi * x[t] .+ Ri * h[t-1] .+ bWi .+ bRi) # input gate
    f[t] = sigm(Wf * x[t] .+ Rf * h[t-1] .+ bWf .+ bRf) # forget gate
    o[t] = sigm(Wo * x[t] .+ Ro * h[t-1] .+ bWo .+ bRo) # output gate
    n[t] = tanh(Wn * x[t] .+ Rn * h[t-1] .+ bWn .+ bRn) # new gate
    c[t] = f[t] .* c[t-1] .+ i[t] .* n[t]               # cell output
    h[t] = o[t] .* tanh(c[t])

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

    rnnforw(r, w, x[, hx, cx]; batchSizes, hy, cy)

Returns a tuple (y,hyout,cyout,rs) given rnn `r`, weights `w`, input
`x` and optionally the initial hidden and cell states `hx` and `cx`
(`cx` is only used in LSTMs).  `r` and `w` should come from a previous
call to `rnninit`.  Both `hx` and `cx` are optional, they are treated
as zero arrays if not provided.  The output `y` contains the hidden
states of the final layer for each time step, `hyout` and `cyout` give
the final hidden and cell states for all layers, `rs` is a buffer the
RNN needs for its gradient calculation.

The boolean keyword arguments `hy` and `cy` control whether `hyout`
and `cyout` will be output.  By default `hy = (hx!=nothing)` and `cy =
(cx!=nothing && r.mode==2)`, i.e. a hidden state will be output if one
is provided as input and for cell state we also require an LSTM.  If
`hy`/`cy` is `false`, `hyout`/`cyout` will be `nothing`. `batchSizes`
can be an integer array that specifies non-uniform batch sizes as
explained below. By default `batchSizes=nothing` and the same batch
size, `size(x,2)`, is used for all time steps.

The input and output dimensions are:

- `x`: (X,[B,T])
- `y`: (H/2H,[B,T])
- `hx`,`cx`,`hyout`,`cyout`: (H,B,L/2L)
- `batchSizes`: `nothing` or `Vector{Int}(T)`

where X is inputSize, H is hiddenSize, B is batchSize, T is seqLength,
L is numLayers.  `x` can be 1, 2, or 3 dimensional.  If
`batchSizes==nothing`, a 1-D `x` represents a single instance, a 2-D
`x` represents a single minibatch, and a 3-D `x` represents a sequence
of identically sized minibatches.  If `batchSizes` is an array of
(non-increasing) integers, it gives us the batch size for each time
step in the sequence, in which case `sum(batchSizes)` should equal
`div(length(x),size(x,1))`. `y` has the same dimensionality as `x`,
differing only in its first dimension, which is H if the RNN is
unidirectional, 2H if bidirectional.  Hidden vectors `hx`, `cx`,
`hyout`, `cyout` all have size (H,B1,L) for unidirectional RNNs, and
(H,B1,2L) for bidirectional RNNs where B1 is the size of the first
minibatch.

"""
function rnnforw{T}(r::RNN, w::KnetArray{T}, x::KnetArray{T},
                    hx::Union{KnetArray{T},Void}=nothing,
                    cx::Union{KnetArray{T},Void}=nothing;
                    handle=cudnnhandle(), training=false,
                    batchSizes=nothing,
                    hy = (hx != nothing),
                    cy = (cx != nothing && r.mode == 2),
                    )

    # Input descriptors
    seqLength = batchSizes==nothing ? size(x,3) : length(batchSizes) # (X,B,T) or (X,B+) with batchSizes
    wDesc = FD3(w)              # (1,1,W)
    xtds = TDs(x,batchSizes)    # (1,X,Bt) x T
    if hx==nothing; hx=hxDesc=C_NULL; else; hxDesc=TD3(hx); end # (H,B,L/2L)
    if cx==nothing || r.mode != 2; cx=cxDesc=C_NULL; else; cxDesc=TD3(cx); end

    # Output arrays and descriptors
    ysize = collect(size(x))
    ysize[1] = r.hiddenSize * (r.direction == 1 ? 2 : 1)
    y = similar(x, ysize...)    # (H/2H,B,T) or (H/2H,B+) -- y mirrors x except for the first dimension
    ytds = TDs(y,batchSizes)    # (1,H/2H,Bt) x T
    
    # Optionally output hidden and cell of last step
    hyout = hyDesc = cyout = cyDesc = C_NULL
    if hy || cy
        firstBatchSize = batchSizes==nothing ? size(x,2) : batchSizes[1]
        hsize = Int[r.hiddenSize, firstBatchSize, r.numLayers * (r.direction == 1 ? 2 : 1)] # (H,B,L/2L)
        if hy; hyout=similar(y,hsize...); hyDesc=TD3(hyout); end
        if cy && r.mode==2; cyout=similar(y,hsize...); cyDesc=TD3(cyout); end
    end

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
              hyDesc, hyout,
              cyDesc, cyout,
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
              hyDesc, hyout,
              cyDesc, cyout,
              ws, wss)
    end
    if hyout == C_NULL; hyout = nothing; end
    if cyout == C_NULL; cyout = nothing; end
    return y, hyout, cyout, rs
end

function rnnforw(::Type{Grad{2}}, dt, t, r, w, x, hx=nothing, cx=nothing; o...)
    y,hy,cy,rs = getval(t)
    dy,dhy,dcy,drs = getval(dt)
    w=getval(w); x=getval(x); hx=getval(hx); cx=getval(cx)
    rnnback(r, w, x, y, dy, hx, cx, dhy, dcy, rs; o...)
end

rnnforw(::Type{Grad{3}}, dt, t, r, w...; o...)=r.dx
rnnforw(::Type{Grad{4}}, dt, t, r, w...; o...)=r.dhx
rnnforw(::Type{Grad{5}}, dt, t, r, w...; o...)=r.dcx

let rnnforw_r = recorder(rnnforw); global rnnforw
    rnnforw(r::RNN, w::Rec, x...; o...)=rnnforw_r(r, w, x...; o..., training=true)
end

function rnnback{T}(r::RNN, w::KnetArray{T}, x::KnetArray{T}, y::KnetArray{T},
                    dy, hx, cx, dhy, dcy, rs; handle=cudnnhandle(), batchSizes=nothing, o...)

    # Input descriptors:
    seqLength = batchSizes==nothing ? size(x,3) : length(batchSizes) # (X,B,T) or (X,B+) with batchSizes
    wDesc = FD3(w)              # (1,1,W)
    xtds = TDs(x,batchSizes)    # (X,B,T) -> (1,X,B) x T
    ytds = TDs(y,batchSizes)    # (H/2H,B,T) -> (1,H/2H,B) x T
    # dytds = TDs(dy,batchSizes)  # we use ytds for dytds
    if dy == nothing; dy=zeros(y); end
    if hx == nothing; hx=hxDesc=C_NULL; else; hxDesc=TD3(hx); end
    if cx == nothing || r.mode != 2; cx=cxDesc=C_NULL; else; cxDesc=TD3(cx); end
    if dhy == nothing; dhy=dhyDesc=C_NULL; else; dhyDesc=TD3(dhy); end
    if dcy == nothing || r.mode != 2; dcy=dcyDesc=C_NULL; else; dcyDesc=TD3(dcy); end

    # Output arrays and descriptors:
    dx = similar(x)             # (X,B,T) or (X,B+) with batchSizes
    # dxtds = TDs(dx,batchSizes)  # we use xtds here
    dw = zeros(w)               # dw is used additively, so we need zeros
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
          ytds, dy,
          dhyDesc, dhy,
          dcyDesc, dcy,
          wDesc, w,
          hxDesc, hx,
          cxDesc, cx,
          xtds, dx,
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


# CPU implementation:
function cudnnGetRNNParams(r::RNN, w::Array; o...)
    layers = r.numLayers * (r.direction == 1 ? 2 : 1)
    ids = r.mode == 2 ? 8 : r.mode == 3 ? 6 : 2
    ws = []; wi = 0
    for m in (true, false)
        for l in 0:layers-1
            for i in 0:ids-1
                # push!(ws, cudnnGetRNNParam(r, w, l, i, m; handle=handle))
                error("wip")
            end
        end
    end
    return ws
end

# CPU version
function rnnforw{T}(r::RNN, w::Array{T}, x::Array{T},
                    hx::Union{Array{T},Void}=nothing,
                    cx::Union{Array{T},Void}=nothing;
                    # handle=cudnnhandle(), training=false,
                    batchSizes=nothing,
                    hy = (hx != nothing),
                    cy = (cx != nothing && r.mode == 2),
                    o...)
    rnntest(r,w,x,hx,cx;batchSizes=batchSizes,hy=hy,cy=cy)
end

# non-CUDNN cpu/gpu version
function rnntest(r::RNN, ws, x, hx=nothing, cx=nothing;
                 batchSizes=nothing,
                 hy = (hx != nothing),
                 cy = (cx != nothing && r.mode == 2),
                 o...)
    if r.direction == 1; error("rnntest bidirectional not implemented yet"); end
    if batchSizes != nothing; error("rnntest batchSizes not implemented yet"); end
    w = cudnnGetRNNParams(r,ws)
    X,B,T = (size(x,i) for i=1:3) # ndims(x) may be 1,2 or 3
    @assert X == r.inputSize
    Y = Int(r.hiddenSize * (r.direction == 1 ? 2 : 1))
    ysize = ntuple(i->(i==1 ? Y : size(x,i)), ndims(x)) # to match ndims(y) to ndims(x)
    H = Int(r.hiddenSize)
    @assert (r.inputMode == 0 || H == X)
    L = Int(r.numLayers * (r.direction == 1 ? 2 : 1))
    hsize = (H,B,L)
    @assert hx == nothing || size(hx) == hsize
    @assert cx == nothing || size(cx) == hsize
    h = hx==nothing ? fill!(similar(x,hsize),0) : hx

    ys = []
    hs = [ h[:,:,l] for l=1:L ]
    if r.mode <= 1
        @assert r.inputMode == 0 || all(w[1:1+r.direction] .== nothing)
        # ht = f(W_i * x_t + R_i h_t-1 + b_Wi + b_Ri)
        f = r.mode == 0 ? relu : tanh
        for t = 1:T
            for l = 1:L
                wx,wh,bx,bh = w[2l-1],w[2l],w[2L+2l-1],w[2L+2l]
                wxt = (l > 1 ? wx' * hs[l-1] : r.inputMode==0 ? wx' * x[:,:,t] : x[:,:,t])
                hs[l] = f.(wxt .+ wh' * hs[l] .+ bx .+ bh)
            end
            push!(ys, hs[L])
        end
    elseif r.mode == 2           # LSTM
        @assert r.inputMode == 0 || all(w[1:4*(1+r.direction)] .== nothing)
        # it = σ(Wixt + Riht-1 + bWi + bRi) 
        # ft = σ(Wfxt + Rfht-1 + bWf + bRf) 
        # ot = σ(Woxt + Roht-1 + bWo + bRo) 
        # c't = tanh(Wcxt + Rcht-1 + bWc + bRc) 
        # ct = ft◦ct-1 + it◦c't 
        # ht = ot◦tanh(ct)
        c = cx==nothing ? fill!(similar(x,hsize),0) : cx
        cs = [ c[:,:,l] for l=1:L ]
        for t = 1:T
            for l = 1:L
                Wi,Wf,Wc,Wo,Ri,Rf,Rc,Ro = w[1+8*(l-1):8l]
                bWi,bWf,bWc,bWo,bRi,bRf,bRc,bRo = w[8L+1+8*(l-1):8L+8l]
                Wixt = (l > 1 ? Wi' * hs[l-1] : r.inputMode==0 ? Wi' * x[:,:,t] : x[:,:,t])
                Wfxt = (l > 1 ? Wf' * hs[l-1] : r.inputMode==0 ? Wf' * x[:,:,t] : x[:,:,t])
                Wcxt = (l > 1 ? Wc' * hs[l-1] : r.inputMode==0 ? Wc' * x[:,:,t] : x[:,:,t])
                Woxt = (l > 1 ? Wo' * hs[l-1] : r.inputMode==0 ? Wo' * x[:,:,t] : x[:,:,t])
                it = sigm.(Wixt .+ Ri' * hs[l] .+ bWi .+ bRi)
                ft = sigm.(Wfxt .+ Rf' * hs[l] .+ bWf .+ bRf)
                ot = sigm.(Woxt .+ Ro' * hs[l] .+ bWo .+ bRo)
                cn = tanh.(Wcxt .+ Rc' * hs[l] .+ bWc .+ bRc)
                cs[l] = ft .* cs[l] .+ it .* cn
                hs[l] = ot .* tanh.(cs[l])
            end
            push!(ys, hs[L])
        end
    elseif r.mode == 3           # GRU
        @assert r.inputMode == 0 || all(w[1:3*(1+r.direction)] .== nothing)
        # rt = σ(Wrxt + Rrht-1 + bWr + bRr)
        # it = σ(Wixt + Riht-1 + bWi + bRu)
        # h't = tanh(Whxt + rt◦(Rhht-1 + bRh) + bWh)
        # ht = (1 - it)◦h't + it◦ht-1
        for t = 1:T
            for l = 1:L
                Wr,Wi,Wh,Rr,Ri,Rh = w[1+6*(l-1):6l]
                bWr,bWi,bWh,bRr,bRi,bRh = w[6L+1+6*(l-1):6L+6l]
                Wrxt = (l > 1 ? Wr' * hs[l-1] : r.inputMode==0 ? Wr' * x[:,:,t] : x[:,:,t])
                Wixt = (l > 1 ? Wi' * hs[l-1] : r.inputMode==0 ? Wi' * x[:,:,t] : x[:,:,t])
                Whxt = (l > 1 ? Wh' * hs[l-1] : r.inputMode==0 ? Wh' * x[:,:,t] : x[:,:,t])
                rt = sigm.(Wrxt .+ Rr' * hs[l] .+ bWr .+ bRr)
                it = sigm.(Wixt .+ Ri' * hs[l] .+ bWi .+ bRi)
                ht = tanh.(Whxt .+ rt .* (Rh' * hs[l] .+ bRh) .+ bWh)
                hs[l] = (1 .- it) .* ht .+ it .* hs[l]
            end
            push!(ys, hs[L])
        end
    else
        error("RNN not supported")
    end
    y = reshape(hcat(ys...), ysize)
    hyout = hy ? reshape(hcat(hs...), hsize) : nothing
    cyout = cy && r.mode == 2 ? reshape(hcat(cs...), hsize) : nothing
    return (y,hyout,cyout,nothing)
end


# We need x[:,:,t] and hx[:,:,l]
using Knet: Index3
import Base: getindex, setindex!
function getindex(A::KnetArray, ::Colon, ::Colon, I::Index3)
    B = reshape(A, stride(A,3), size(A,3))
    reshape(B[:,I], size(A,1), size(A,2))
end
function setindex!(x::KnetArray, y, ::Colon, ::Colon, I::Index3)
    reshape(x, stride(x,3), size(x,3))[:,I] = y
    return x
end
function getindex{T,I<:Integer}(x::KnetArray{T,2}, ::Colon, m::Array{I,2})
    reshape(x[:,vec(m)], size(x,1), size(m,1), size(m,2))
end
