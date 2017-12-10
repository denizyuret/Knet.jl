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
    inputMode::Cint    # CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT = 1    
    direction::Cint    # CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL = 1
    mode::Cint         # CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3
    algo::Cint         # CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
    dataType::DataType # CUDNN_DATA_FLOAT  = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_DATA_HALF   = 2
    rnnDesc::Union{RD,Void}
    dropoutDesc::Union{DD,Void}
    dx
    dhx
    dcx
end

rnnids(r) = (r.mode == 2 ? 8 : r.mode == 3 ? 6 : 2)

function cudnnGetRNNParamsSize(r::RNN; handle=cudnnhandle())
    if r.rnnDesc != nothing
        res = Csize_t[0]
        xDesc = TD(r.dataType, 1, r.inputSize, 1)    # xDesc: (1,X,B) where X = inputSize, B is ignored, so assume 1
        @cuda(cudnn, cudnnGetRNNParamsSize,
              # handle, rnndesc, xdesc, result, dataType
              (Cptr,  Cptr, Cptr, Ptr{Csize_t}, UInt32),
              handle, r.rnnDesc, xDesc, res, DT(r.dataType))
        div(res[1], sizeof(r.dataType))
    else # on CPU, so we guess the size
        X,H,L,I = r.inputSize, r.hiddenSize, r.numLayers, rnnids(r)
        biases = L*I
        inputMatrices = (r.inputMode == 1 ? 0 : r.direction == 1 ? I : div(I,2))
        hiddenMatrices = (r.direction == 1 ? (L-1)*I : (L-1)*I + div(I,2))
        biases * H + inputMatrices * X * H + hiddenMatrices * H * H
    end
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

function TDs{A}(x::KnetArray{A},batchSizes) # x: (X,B*), batchSizes gives us Bt sizes
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


# call gpu everytime to reflect device changes
gethandle() = gpu() >= 0 ? cudnnhandle() : nothing


"""

        rnnparam{T}(r::RNN, w::KnetArray{T}, layer, id, param)

    Return a single weight matrix or bias vector as a slice of w.

    Valid `layer` values:
    * For unidirectional RNNs 1:numLayers
    * For bidirectional RNNs 1:2*numLayers, forw and back layers alternate.

    Valid `id` values:
    * For RELU and TANH RNNs, input = 1, hidden = 2.
    * For GRU reset = 1,4; update = 2,5; newmem = 3,6; 1:3 for input, 4:6 for hidden
    * For LSTM inputgate = 1,5; forget = 2,6; newmem = 3,7; output = 4,8; 1:4 for input, 5:8 for hidden

    Valid `param` values:
    * Return the weight matrix (transposed!) if `param==1`.
    * Return the bias vector if `param==2`.

    The effect of skipInput: Let I=1 for RELU/TANH, 1:3 for GRU, 1:4 for LSTM
    * For skipInput=false (default), rnnparam(r,w,1,I,1) is a (inputSize,hiddenSize) matrix.
    * For skipInput=true, rnnparam(r,w,1,I,1) is `nothing`.
    * For bidirectional, the same applies to rnnparam(r,w,2,I,1): the first back layer.

    """
function rnnparam(r::RNN, w, layer::Integer, id::Integer, par::Integer; handle=gethandle(), useview=false)
    # w could be a Rec, KnetArray, or Array so typing w::KnetArray{T} is not an option
    ((1 <= par <= 2) &&
     ((r.direction == 0 && 1 <= layer <= r.numLayers) ||
      (r.direction == 1 && 1 <= layer <= 2*r.numLayers)) &&
     ((r.mode == 0 && 1 <= id <= 2) ||
      (r.mode == 1 && 1 <= id <= 2) ||
      (r.mode == 2 && 1 <= id <= 8) ||
      (r.mode == 3 && 1 <= id <= 6))) || error("Bad parameter index")
    if isa(getval(w), KnetArray)
        T = eltype(w)
        xDesc = TD(T,1,r.inputSize,1)
        wDesc = FD(T,1,1,length(w))
        paramDesc = FD(T,1,1,1,1)
        param = Cptr[0]
        if par == 1 # matrix
            @cuda(cudnn, cudnnGetRNNLinLayerMatrixParams,
                  (Cptr, Cptr, Cint, #handle,rdesc, layer
                   Cptr, Cptr, Cptr, #xDesc, wDesc, w
                   Cint, Cptr, Ptr{Cptr}), #lid, lmatdesc, linlayermat
                  handle, r.rnnDesc, layer-1,
                  xDesc, wDesc, getval(w),
                  id-1, paramDesc, param)
        else # bias
            @cuda(cudnn, cudnnGetRNNLinLayerBiasParams,
                  (Cptr, Cptr, Cint, #handle,rdesc, layer
                   Cptr, Cptr, Cptr, #xDesc, wDesc, w
                   Cint, Cptr, Ptr{Cptr}), #lid, lmatdesc, linlayermat
                  handle, r.rnnDesc, layer-1,
                  xDesc, wDesc, getval(w),
                  id-1, paramDesc, param)
        end
        dt,sz = cudnnGetFilterNdDescriptor(paramDesc)
        len = prod(sz)
        i1 = 1 + div(Int(param[1] - pointer(w)), sizeof(T))
        i2 = i1 + len - 1
    else
        # guess i1,i2,len from layer,id,par and rnn specs
        ids = rnnids(r)
        if par == 1 # matrix
            # input matrices are (inputSize,hiddenSize) all others (hiddenSize,hiddenSize)
            # if inputMode==1 then input matrices are empty
            # I=1 for RELU/TANH, 1:3 for GRU, 1:4 for LSTM are input matrices with L=1 for uni and L=1,2 for bi
            inputLayer(r,l)=(l==1 || (l==2 && r.direction==1))
            X, H = r.inputSize, r.hiddenSize
            XH, HH = X*H, H*H
            inputIds = div(ids,2)
            i1 = i2 = len = 0
            for l = 1:layer, i = 1:ids
                if inputLayer(r,l) && i <= inputIds
                    len = (r.inputMode==0 ? XH : 0)
                elseif l>2 && r.direction==1 && i <= div(ids, 2)
                    # bidirectional weight
                    len = 2HH
                else
                    len = HH
                end
                i2 += len
                if l==layer && i==id; break; end
            end
            if len==0; len=1; end # cudnn uses size (1,1,1) for empty weights
            i1 = i2 - len + 1
        else # bias
            # all biases are length=hidden and there are always numLayers * numIds of them
            len = r.hiddenSize
            layers = r.numLayers * (r.direction == 1 ? 2 : 1)
            i1 = 1 + length(w) - layers * ids * len + ((id-1) + ids * (layer-1)) * len
            i2 = i1 + len - 1
        end
    end
    @inline access(a, rng) = (isa(a, KnetArray) || ~useview) ? a[rng] : view(a, rng)
    if len == 1 # empty weights when inputMode=1 show up as size (1,1,1)
        nothing
    elseif par == 1 # matrix
        h = Int(r.hiddenSize)
        reshape(access(w, i1:i2),
                (div(len,h),h)) # weight matrices are transposed
    else # bias
        access(w, i1:i2)
    end
end


"""

        rnnparams(r::RNN, w)

    Split w into individual parameters and return them as an array.

    The order of params returned (subject to change):
    * All weight matrices come before all bias vectors.
    * Matrices and biases are sorted lexically based on (layer,id).
    * See @doc rnnparam for valid layer and id values.
    * Input multiplying matrices are `nothing` if r.inputMode = 1.

    """
function rnnparams(r::RNN, w; handle=gethandle(), useview=false)
    layers = r.numLayers * (r.direction == 1 ? 2 : 1)
    ids = rnnids(r)
    ws = []
    for m in (1,2)
        for l in 1:layers
            for i in 1:ids
                push!(ws, rnnparam(r, w, l, i, m; handle=handle, useview=useview))
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
    - `binit=zeros`: Weight initialization method for bias vectors.
    - `usegpu=(gpu()>=0): GPU used by default if one exists.

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
                 handle=gethandle(),
                 numLayers=1,
                 dropout=0.0,
                 skipInput=false,     # CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT = 1
                 bidirectional=false, # CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL = 1
                 rnnType=:lstm,       # CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3
                 dataType=Float32,    # CUDNN_DATA_FLOAT  = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_DATA_HALF   = 2
                 algo=0,              # CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
                 seed=0,              # seed=0 for random init, positive integer for replicability
                 winit=xavier,
                 binit=zeros,
                 usegpu=(gpu()>=0),
                 )
    inputMode = skipInput ? 1 : 0
    direction = bidirectional ? 1 : 0
    mode = findfirst((:relu,:tanh,:lstm,:gru), rnnType) - 1
    if mode < 0; error("rnninit: Valid modes are :relu,:tanh,:lstm,:gru"); end
    if usegpu
        rnnDesc = RD()
        dropoutDesc = DD(handle=handle,dropout=dropout,seed=seed) # Need to keep dropoutDesc in RNN so it does not get gc'ed.
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
        w = KnetArray{dataType}(1,1,cudnnGetRNNParamsSize(r))
    else
        r = RNN(inputSize,hiddenSize,numLayers,dropout,inputMode,direction,mode,algo,dataType,nothing,nothing,nothing,nothing,nothing)
        # TODO: make this a separate function?
        w = begin
            whidden = hiddenSize * hiddenSize
            winput =  skipInput ? 0 : hiddenSize * inputSize
            bhidden = hiddenSize
            binput =  bhidden
            coef = (mode == 2 ? 4 : mode == 3 ? 3 : 1) * (1 + direction)
            nparams = 0
            for i = 1:r.numLayers
                nparams += coef * (whidden + winput + bhidden + binput)
                winput = (1 + direction) * whidden
                binput = bhidden
            end
            Array{dataType}(1,1,nparams)
        end
    end
    for a in rnnparams(r,w; handle=handle, useview=true)
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

rnnforw(r::Rec{RNN}, w...; o...)=rnnforw(getval(r), w...; o...)

let rnnforw_r = recorder(rnnforw); global rnnforw
    rnnforw{K<:KnetArray}(r::RNN, w::Rec{K}, x...; o...)=rnnforw_r(r, w, x...; o..., training=true)
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
    #if r.direction == 1; error("rnntest bidirectional not implemented yet"); end
    if r.dropout != 0; error("rnntest dropout not implemented yet"); end
    if batchSizes != nothing; error("rnntest batchSizes not implemented yet"); end
    w = rnnparams(r,ws)
    X,B,T = (size(x,i) for i=1:3) # ndims(x) may be 1,2 or 3
    @assert X == r.inputSize
    Y = Int(r.hiddenSize * (r.direction == 1 ? 2 : 1))
    ysize = ntuple(i->(i==1 ? Y : size(x,i)), ndims(x)) # to match ndims(y) to ndims(x)
    H = Int(r.hiddenSize)
    #@assert (r.inputMode == 0 || H == X)
    L = Int(r.numLayers) * (r.direction == 1 ? 2 : 1)
    hsize = (H, B, L)
    @assert hx == nothing || size(hx) == hsize
    @assert cx == nothing || size(cx) == hsize
    h = hx==nothing ? fill!(similar(x,hsize),0) : hx
    hs = [ h[:,:,l] for l=1:L ]
    ys = []
    direction = r.direction
    #=
    All complexity of bidirectional execution
    is packed inside this inline function.
    This causes code repetition, but  works w/o
    touching the existing unidirectional test code
    =#
    @inline bidirect(update_h!) = begin
        X = x
        for l = 1:(1+direction):L
            skip = l==1 && r.inputMode==1
            hts = []
            for t = 1:T
                for (i,ti) in zip([l, l+1], [t, T-t+1])
                    # this function updates h[i]
                    update_h!(X, i, ti, skip) 
                    push!(hts, hs[i])
                end
            end
            # construct the next layer output
            yforw = Array{Any}(hts[1:2:end-1])
            yback = Array{Any}(reverse(hts[2:2:end]))
            ybs = []
            for (yf, yb) in zip(yforw, yback)
                push!(ybs, vcat(yf, yb))
            end
            # now ybs contans (2 * hiddenSize, batchSize) matrices
            # so cat them to add time dimension
            X = reshape(hcat(ybs...), (2r.hiddenSize, size(x,2), length(ybs)))
        end
        ys = X
    end
    
    if r.mode <= 1
        #@assert r.inputMode == 0 || all(w[1:1+r.direction] .== nothing)
        f = r.mode == 0 ? relu : tanh
        if direction == 0
            for t = 1:T
                for l = 1:L
                    wx,wh,bx,bh = w[2l-1],w[2l],w[2L+2l-1],w[2L+2l]
                    wxt = (l > 1 ? wx' * hs[l-1] : r.inputMode==0 ? wx' * x[:,:,t] : x[:,:,t])
                    hs[l] = f.(wxt .+ wh' * hs[l] .+ bx .+ bh)
                end
                push!(ys, hs[L])
            end
        else
            bidirect() do X, i, ti, skip
                wx,wh,bx,bh = w[2i-1],w[2i],w[2L+2i-1],w[2L+2i]
                wxt =  skip ? X[:,:,ti] : wx' * X[:,:,ti]
                hs[i] = f.(wxt .+ wh' * hs[i] .+ bx .+ bh)
            end
        end
    elseif r.mode == 2           # LSTM
        #@assert r.inputMode == 0 || all(w[1:4*(1+r.direction)] .== nothing)
        # it = σ(Wixt + Riht-1 + bWi + bRi) 
        # ft = σ(Wfxt + Rfht-1 + bWf + bRf) 
        # ot = σ(Woxt + Roht-1 + bWo + bRo) 
        # c't = tanh(Wcxt + Rcht-1 + bWc + bRc) 
        # ct = ft◦ct-1 + it◦c't 
        # ht = ot◦tanh(ct)
        c = cx==nothing ? fill!(similar(x,hsize),0) : cx
        cs = [ c[:,:,l] for l=1:L ]
        if direction == 0
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
        else
            bidirect() do X, i, ti, skip
                Wi,Wf,Wc,Wo,Ri,Rf,Rc,Ro = w[1+8*(i-1):8i]
                bWi,bWf,bWc,bWo,bRi,bRf,bRc,bRo = w[8L+1+8*(i-1):8L+8i]
                Wixt = skip ? X[:,:,ti] : Wi' * X[:,:,ti]
                Wfxt = skip ? X[:,:,ti] : Wf' * X[:,:,ti]
                Wcxt = skip ? X[:,:,ti] : Wc' * X[:,:,ti]
                Woxt = skip ? X[:,:,ti] : Wo' * X[:,:,ti]
                it = sigm.(Wixt .+ Ri' * hs[i] .+ bWi .+ bRi)
                ft = sigm.(Wfxt .+ Rf' * hs[i] .+ bWf .+ bRf)
                ot = sigm.(Woxt .+ Ro' * hs[i] .+ bWo .+ bRo)
                cn = tanh.(Wcxt .+ Rc' * hs[i] .+ bWc .+ bRc)
                cs[i] = ft .* cs[i] .+ it .* cn
                hs[i] = ot .* tanh.(cs[i])
            end
        end
    elseif r.mode == 3           # GRU
        #@assert r.inputMode == 0 || all(w[1:3*(1+r.direction)] .== nothing)
        # rt = σ(Wrxt + Rrht-1 + bWr + bRr)
        # it = σ(Wixt + Riht-1 + bWi + bRu)
        # h't = tanh(Whxt + rt◦(Rhht-1 + bRh) + bWh)
        # ht = (1 - it)◦h't + it◦ht-1
        if direction == 0
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
            bidirect() do X, i, ti, skip
                Wr,Wi,Wh,Rr,Ri,Rh = w[1+6*(i-1):6i]
                bWr,bWi,bWh,bRr,bRi,bRh = w[6L+1+6*(i-1):6L+6i]
                Wrxt = skip ? X[:, :, ti] : Wr' * X[:, :, ti]
                Wixt = skip ? X[:, :, ti] : Wi' * X[:, :, ti]
                Whxt = skip ? X[:, :, ti] : Wh' * X[:, :, ti]
                rt = sigm.(Wrxt .+ Rr' * hs[i] .+ bWr .+ bRr)
                it = sigm.(Wixt .+ Ri' * hs[i] .+ bWi .+ bRi)
                ht = tanh.(Whxt .+ rt .* (Rh' * hs[i] .+ bRh) .+ bWh)
                hs[i] = (1 .- it) .* ht .+ it .* hs[i]
            end
        end
    else
        error("RNN not supported")
    end
    y = r.direction == 0 ? reshape(hcat(ys...), ysize) : ys
    hyout = hy ? reshape(hcat(hs...), hsize) : nothing
    cyout = cy && r.mode == 2 ? reshape(hcat(cs...), hsize) : nothing
    return (y,hyout,cyout,nothing)
end


# Hack for JLD file load/save of RNNs:
if Pkg.installed("JLD") != nothing
    import JLD: writeas, readas
    type RNNJLD; inputSize; hiddenSize; numLayers; dropout; inputMode; direction; mode; algo; dataType; end
    writeas(r::RNN) = RNNJLD(r.inputSize, r.hiddenSize, r.numLayers, r.dropout, r.inputMode, r.direction, r.mode, r.algo, r.dataType)
    readas(r::RNNJLD) = rnninit(r.inputSize, r.hiddenSize, numLayers=r.numLayers, dropout=r.dropout, skipInput=(r.inputMode==1), bidirectional=(r.direction==1), rnnType=(:relu,:tanh,:lstm,:gru)[1+r.mode], algo=r.algo, dataType=r.dataType)[1]
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

