using CUDA

# TODO: finish cpu implementation.

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

"""
    rnn = RNN(inputSize, hiddenSize; opts...)
    rnn(x; batchSizes) => y
    rnn.h, rnn.c  # hidden and cell states

`RNN` returns a callable RNN object `rnn`. Given a minibatch of sequences `x`, `rnn(x)`
returns `y`, the hidden states of the final layer for each time step. `rnn.h` and `rnn.c`
fields can be used to set the initial hidden states and read the final hidden states of all
layers.  Note that the final time step of `y` always contains the final hidden state of the
last layer, equivalent to `rnn.h` for a single layer network.

**Dimensions:** The input `x` can be 1, 2, or 3 dimensional and `y` will have the same number
of dimensions as `x`. size(x)=(X,[B,T]) and size(y)=(H/2H,[B,T]) where X is inputSize, B is
batchSize, T is seqLength, H is hiddenSize, 2H is for bidirectional RNNs. By default a 1-D `x`
represents a single instance for a single time step, a 2-D `x` represents a single minibatch
for a single time step, and a 3-D `x` represents a sequence of identically sized minibatches
for multiple time steps. The output `y` gives the hidden state (of the final layer for
multi-layer RNNs) for each time step. The fields `rnn.h` and `rnn.c` represent the hidden
states of all layers in a single time step and have size (H,B,L/2L) where L is numLayers and
2L is for bidirectional RNNs.

**batchSizes:** If `batchSizes=nothing` (default), all sequences in a minibatch are assumed to
be the same length. If `batchSizes` is an array of (non-increasing) integers, it gives us the
batch size for each time step (allowing different sequences in the minibatch to have different
lengths). In this case `x` will typically be 2-D with the second dimension representing
variable size batches for time steps. If `batchSizes` is used, `sum(batchSizes)` should equal
`length(x) ÷ size(x,1)`. When the batch size is different in every time step, hidden states
will have size (H,B,L/2L) where B is always the size of the first (largest) minibatch.

**Hidden states:** The hidden and cell states are kept in `rnn.h` and `rnn.c` fields (the cell
state is only used by LSTM). They can be initialized during construction using the `h` and `c`
keyword arguments, or modified later by direct assignment. Valid values are `nothing`
(default), `0`, or an array of the right type and size possibly wrapped in a `Param`. If the
value is `nothing` the initial state is assumed to be zero and the final state is discarded
keeping the value `nothing`. If the value is `0` the initial state is assumed to be zero and
`0` is replaced by the final state on return. If the value is a valid state, it is used as the
initial state and is replaced by the final state on return.

In a differentiation context the returned final hidden states will be wrapped in `Result`
types. This is necessary if the same RNN object is to be called multiple times in a single
iteration. Between iterations (i.e. after diff/update) the hidden states need to be unboxed
with e.g. `rnn.h = value(rnn.h)` to prevent spurious dependencies. This happens automatically
during the backward pass for GPU RNNs but needs to be done manually for CPU RNNs. See the
[CharLM Tutorial](https://github.com/denizyuret/Knet.jl/blob/master/tutorial/80.charlm.ipynb)
for an example.

**Keyword arguments for RNN:**
- `h=nothing`: Initial hidden state.
- `c=nothing`: Initial cell state.
- `rnnType=:lstm` Type of RNN: One of :relu, :tanh, :lstm, :gru.
- `numLayers=1`: Number of RNN layers.
- `bidirectional=false`: Create a bidirectional RNN if `true`.
- `dropout=0`: Dropout probability. Applied to input and between layers.
- `skipInput=false`: Do not multiply the input with a matrix if `true`.
- `dataType=Float32`: Data type to use for weights.
- `algo=0`: Algorithm to use, see CUDNN docs for details.
- `seed=0`: Random number seed for dropout. Uses `time()` if 0.
- `winit=xavier_uniform`: Weight initialization method for matrices.
- `binit=zeros`: Weight initialization method for bias vectors.
- `finit=ones`: Weight initialization method for the bias of forget gates.
- `usegpu=(gpu()>=0)`: GPU used by default if one exists.

**Formulas:** RNNs compute the output h[t] for a given iteration from the recurrent input
h[t-1] and the previous layer input x[t] given matrices W, R and biases bW, bR from the
following equations:

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
RNN


"RNN descriptor"
mutable struct RD; ptr; end

"Dropout descriptor"
mutable struct DD; ptr; states::KnetArray{UInt8,1}; end

mutable struct RNN
    w
    h
    c
    inputSize::Cint
    hiddenSize::Cint
    numLayers::Cint
    dropout::Float64
    seed::Culonglong
    inputMode::Cint    # CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT = 1
    direction::Cint    # CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL = 1
    mode::Cint         # CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3
    algo::Cint         # CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
    dataType::DataType # CUDNN_DATA_FLOAT  = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_DATA_HALF   = 2
    rnnDesc::Union{RD,Nothing}
    dropoutDesc::Union{DD,Nothing}
    dx
    dhx
    dcx
end

function RNN(inputSize, hiddenSize;
             h=nothing, c=nothing,
             handle=gethandle(),
             numLayers=1,
             dropout=0.0,
             skipInput=false,     # CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT = 1
             bidirectional=false, # CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL = 1
             rnnType=:lstm,       # CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3
             dataType=eltype(atype()),    # CUDNN_DATA_FLOAT  = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_DATA_HALF   = 2
             algo=0,              # CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
             seed=0,              # seed=0 for random init, positive integer for replicability
             winit=xavier_uniform,
             binit=zeros,
             finit=ones,        # forget bias for lstm
             usegpu=(gpu()>=0 && atype() <: KnetArray),
             )
    w = dx = dhx = dcx = nothing
    inputSize = Cint(inputSize)
    hiddenSize = Cint(hiddenSize)
    numLayers = Cint(numLayers)
    dropout = Float64(dropout)
    seed = Culonglong(seed)
    inputMode = Cint(skipInput ? 1 : 0)
    direction = Cint(bidirectional ? 1 : 0)
    mode = findfirst(isequal(rnnType), (:relu,:tanh,:lstm,:gru))
    @assert mode !== nothing "RNN: Valid modes are :relu,:tanh,:lstm,:gru"
    mode = Cint(mode - 1)
    algo = Cint(algo)
    @assert dataType isa DataType
    dropoutDesc = usegpu ? DD(handle=handle,dropout=dropout,seed=seed) : nothing # Need to keep dropoutDesc in RNN so it does not get gc'ed.
    rnnDesc = usegpu ? RD(hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,dataType) : nothing
    r = RNN(w,h,c,inputSize,hiddenSize,numLayers,dropout,seed,inputMode,direction,mode,algo,dataType,rnnDesc,dropoutDesc,dx,dhx,dcx)
    r.w = Array{dataType}(undef,1,1,getRNNParamsSize(r))
    for a in rnnparams(r; handle=handle, useview=true)
        if a == nothing
            continue
        elseif ndims(a) == 2
            copyto!(a, winit(dataType, size(a)))
        elseif ndims(a) == 1
            copyto!(a, binit(dataType, size(a)))
        else
            error("Invalid RNN param $(summary(a))")
        end
    end
    if rnnType == :lstm         # separate initialization for lstm forget biases
        for layer in 1:(numLayers*(bidirectional ? 2 : 1)), id in (2,6), param in (2,)
            a = rnnparam(r, layer, id, param, useview=true, handle=handle)
            if a != nothing
                copyto!(a, finit(dataType, size(a)))
            end
        end
    end
    # many copyto! ops to gpu is expensive (~20s), so we init on cpu and copy it over once
    r.w = Param(usegpu ? KnetArray(r.w) : r.w)
    return r
end

function (r::RNN)(x; batchSizes=nothing)
    # Check type/dims of inputs
    WTYPE = typeof(vec(value(r.w)))
    @assert length(x) > 0
    @assert vec(value(x)) isa WTYPE
    @assert ndims(x) <= 3
    @assert size(x,1) == r.inputSize
    HSIZE = (r.hiddenSize, batchSizes == nothing ? size(x,2) : batchSizes[1], r.numLayers * (r.direction + 1))
    @assert r.h == nothing || r.h == 0 || (vec(value(r.h)) isa WTYPE && ndims(r.h) <= 3 && (size(r.h,1),size(r.h,2),size(r.h,3)) == HSIZE)
    @assert r.c == nothing || r.c == 0 || (vec(value(r.c)) isa WTYPE && ndims(r.c) <= 3 && (size(r.c,1),size(r.c,2),size(r.c,3)) == HSIZE)
    # apply dropout to input: rnnforw only applies it between layers.
    x = dropout(x,r.dropout)
    # use hidden inputs unless nothing or 0
    hx = (r.h == 0 ? nothing : r.h)
    cx = (r.c == 0 ? nothing : r.c)
    # produce hidden outputs if hidden inputs != nothing
    hy = (r.h != nothing)
    cy = (r.mode == 2 && r.c != nothing)
    (y, hyout, cyout, rs) = rnnforw(r, r.w, x, hx, cx; hy=hy, cy=cy, batchSizes=batchSizes)
    # In a @diff context hyout and cyout will be Results
    # This is necessary to keep dependencies if r is run several times during one forw pass
    # However unless Results are cleared up before next iteration we will run out of memory
    # I do the clearing in the back function when I know the iteration is done for sure.
    if hy; r.h = hyout; end
    if cy; r.c = cyout; end
    return y
end

function show(io::IO, r::RNN)
    print(io, ("RNNRELU","RNNTANH","LSTM","GRU")[r.mode+1], "(input=", r.inputSize, ",hidden=", r.hiddenSize)
    if r.direction == 1; print(io, ",bidirectional"); end
    if r.numLayers > 1; print(io, ",layers=", r.numLayers); end
    if r.dropout > 0; print(io, ",dropout=", r.dropout); end
    if r.inputMode == 1; print(io, ",skipinput"); end
    if r.dataType != Float32; print(io, ',', r.dataType); end
    print(io, ')')
end



Base.unsafe_convert(::Type{Cptr}, dd::DD)=dd.ptr

function DD(; handle=CUDNN.handle(), dropout=0.0, seed=0, o...)
    if seed==0; seed=floor(Culonglong,time()); end
    d = Cptr[0]; s = Csize_t[0] # TODO: Can multiple RNNs share dropout descriptors? Can dropout probability be changed?
    #@cudnn(cudnnCreateDropoutDescriptor,(Ptr{Cptr},),d)
    CUDNN.cudnnCreateDropoutDescriptor(d)
    #@cudnn(cudnnDropoutGetStatesSize,(Cptr,Ptr{Csize_t}),handle,s)
    CUDNN.cudnnDropoutGetStatesSize(handle,s)
    states = KnetArray{UInt8}(undef,s[1]) # TODO: Can this be shared? 638976 bytes.
    # @cudnn(cudnnSetDropoutDescriptor,(Cptr,Cptr,Cfloat,Cptr,Csize_t,Culonglong),
    #       d[1],handle,dropout,states,bytes(states),seed)
    CUDNN.cudnnSetDropoutDescriptor(d[1],handle,dropout,states,bytes(states),seed)
    dd = DD(d[1],states)
    # finalizer(x->@cudnn(cudnnDestroyDropoutDescriptor,(Cptr,),x.ptr),dd)
    finalizer(x->CUDNN.cudnnDestroyDropoutDescriptor(x.ptr),dd)
    return dd
end


Base.unsafe_convert(::Type{Cptr}, rd::RD)=rd.ptr

function RD()
    d = Cptr[0]
    # @cudnn(cudnnCreateRNNDescriptor,(Ptr{Cptr},),d)
    CUDNN.cudnnCreateRNNDescriptor(d)
    rd = RD(d[1])
    #finalizer(x->@cudnn(cudnnDestroyRNNDescriptor,(Cptr,),x.ptr),rd)
    finalizer(x->CUDNN.cudnnDestroyRNNDescriptor(x.ptr),rd)
    return rd
end

function RD(hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,dataType; handle=gethandle())
    rnnDesc = RD()
    # if cudnnVersion >= 7000
    #     @cudnn(cudnnSetRNNDescriptor,(Cptr,Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint,Cint),
    #           handle,rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,DT(dataType))
    # elseif cudnnVersion >= 6000
    #     @cudnn(cudnnSetRNNDescriptor_v6,(Cptr,Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint,Cint),
    #           handle,rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,DT(dataType))
    # elseif cudnnVersion >= 5000
    #     @cudnn(cudnnSetRNNDescriptor,(Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint),
    #           rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,DT(dataType))
    # else
    #     error("CUDNN $cudnnVersion does not support RNNs")
    # end
    inputMode = CUDNN.cudnnRNNInputMode_t(inputMode)
    direction = CUDNN.cudnnDirectionMode_t(direction)
    mode = CUDNN.cudnnRNNMode_t(mode)
    algo = CUDNN.cudnnRNNAlgo_t(algo)
    dt = CUDNN.cudnnDataType_t(DT(dataType))
    CUDNN.cudnnSetRNNDescriptor(handle,rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,dt)
    return rnnDesc
end

rnnids(r) = (r.mode == 2 ? 8 : r.mode == 3 ? 6 : 2)

function cudnnGetRNNParamsSize(r::RNN; handle=CUDNN.handle())
    res = Csize_t[0]
    xDesc = TD(r.dataType, 1, r.inputSize, 1)    # xDesc: (1,X,B) where X = inputSize, B is ignored, so assume 1
    # @cudnn(cudnnGetRNNParamsSize,
    #        # handle, rnndesc, xdesc, result, dataType
    #        (Cptr,  Cptr, Cptr, Ptr{Csize_t}, UInt32),
    #        handle, r.rnnDesc, xDesc, res, DT(r.dataType))
    dt = CUDNN.cudnnDataType_t(DT(r.dataType))
    CUDNN.cudnnGetRNNParamsSize(handle, r.rnnDesc, xDesc, res, dt)
    div(res[1], sizeof(r.dataType))
end

# This is buggy, why?
# X,H,L,I = r.inputSize, r.hiddenSize, r.numLayers, rnnids(r)
# biases = L*I
# inputMatrices = (r.inputMode == 1 ? 0 : r.direction == 1 ? I : div(I,2))
# hiddenMatrices = (r.direction == 1 ? (L-1)*I : (L-1)*I + div(I,2))
# biases * H + inputMatrices * X * H + hiddenMatrices * H * H

function getRNNParamsSize(r::RNN)
    whidden = r.hiddenSize * r.hiddenSize
    winput =  r.inputMode == 1 ? 0 : r.hiddenSize * r.inputSize
    bhidden = r.hiddenSize
    binput =  bhidden
    coef = (r.mode == 2 ? 4 : r.mode == 3 ? 3 : 1) * (1 + r.direction)
    nparams = 0
    for i = 1:r.numLayers
        nparams += coef * (whidden + winput + bhidden + binput)
        winput = (1 + r.direction) * whidden
        binput = bhidden
    end
    if r.rnnDesc != nothing
        @assert nparams == cudnnGetRNNParamsSize(r)
    end
    return nparams
end

"Keeps an array of 3D tensor descriptors"
mutable struct TDs; pvec::Vector{Cptr}; xDesc::Vector{TD}; end     # Keep xDesc in TDs so it does not get gc'ed

Base.unsafe_convert(::Type{Ptr{Cptr}}, tds::TDs)=pointer(tds.pvec)
Base.length(tds::TDs)=length(tds.pvec)

function TDs(x::KnetArray{A},::Nothing) where {A} # Treat x: (X,B?,T?) as a 4D array: (1,X,B,T)
    xDesc = TD(A,1,size(x,1),size(x,2)) # we can use a single xDesc
    pvec = Vector{Cptr}(undef, size(x,3))
    pvec[:] .= xDesc.ptr
    return TDs(pvec, [xDesc])
end

function TDs(x::KnetArray{A},batchSizes) where {A} # x: (X,B*), batchSizes gives us Bt sizes
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

function cudnnGetRNNWorkspaceSize(rd::RD, tds::TDs; handle=CUDNN.handle())
    res = Csize_t[1]
    # @cudnn(cudnnGetRNNWorkspaceSize,
    #       # handle, rnndesc, seqLength, xdesc, res        ,
    #       (Cptr,  Cptr, Cint, Ptr{Cptr}, Ptr{Csize_t}),
    #       handle, rd, length(tds), tds, res)
    CUDNN.cudnnGetRNNWorkspaceSize(handle, rd, length(tds), tds, res)
    return Int(res[1])
end

function cudnnGetRNNTrainingReserveSize(rd::RD, tds::TDs; handle=CUDNN.handle())
    res = Csize_t[1]
    # @cudnn(cudnnGetRNNTrainingReserveSize,
    #       # handle, rnndesc, seqLength, xdesc, res        ,
    #       (Cptr,  Cptr, Cint, Ptr{Cptr}, Ptr{Csize_t}),
    #       handle, rd, length(tds), tds, res)
    CUDNN.cudnnGetRNNTrainingReserveSize(handle, rd, length(tds), tds, res)
    return Int(res[1])
end

# Return eltype,size
function cudnnGetFilterNdDescriptor(wDesc::FD; nbDimsRequested = 8)
    dataType = Cint[0]
    format = Cint[0]
    nbDims = Cint[0]
    filterDimA = Vector{Cint}(undef,nbDimsRequested)
    # @cudnn(cudnnGetFilterNdDescriptor,
    #       (Cptr, Cint, Ptr{UInt32}, Ptr{UInt32}, Ptr{Cint}, Ptr{Cint}),
    #       wDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)
    CUDNN.cudnnGetFilterNdDescriptor(wDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)
    if nbDims[1] > nbDimsRequested
        cudnnGetFilterNdDescriptor(wDesc::FD; nbDimsRequested = nbDims[1])
    else
        (Float32,Float64,Float16)[1+dataType[1]],
        (filterDimA[nbDims[1]:-1:1]...,)
    end
end


# call gpu everytime to reflect device changes
gethandle() = gpu() >= 0 ? CUDNN.handle() : nothing


"""
    rnnparam(r::RNN, layer, id, param)

Return a single weight matrix or bias vector as a slice of RNN weights.

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
* For skipInput=false (default), rnnparam(r,1,I,1) is a (inputSize,hiddenSize) matrix.
* For skipInput=true, rnnparam(r,1,I,1) is `nothing`.
* For bidirectional, the same applies to rnnparam(r,2,I,1): the first back layer.
* The input biases (par=2) are returned even if skipInput=true.
"""
function rnnparam(r::RNN, layer::Integer, id::Integer, par::Integer; handle=gethandle(), useview=false)
    params_are_good = 
    ((1 <= par <= 2) &&
     ((r.direction == 0 && 1 <= layer <= r.numLayers) ||
      (r.direction == 1 && 1 <= layer <= 2*r.numLayers)) &&
     ((r.mode == 0 && 1 <= id <= 2) ||
      (r.mode == 1 && 1 <= id <= 2) ||
      (r.mode == 2 && 1 <= id <= 8) ||
      (r.mode == 3 && 1 <= id <= 6)))
    params_are_good || throw(ArgumentError("Bad arguments for rnnparam, please see doc."))
    should_return_nothing =
        ((r.inputMode == 1) &&
         (par == 1) &&
         ((r.mode == 0 && id == 1) ||
          (r.mode == 1 && id == 1) ||
          (r.mode == 2 && 1 <= id <= 4) ||
          (r.mode == 3 && 1 <= id <= 3)) &&
         ((layer == 1) ||
          (r.direction == 1 && layer == 2)))

    i1 = i2 = len = 0
    w = value(r.w)
    if isa(w, KnetArray)
        T = eltype(w)
        xDesc = TD(T,1,r.inputSize,1)
        wDesc = FD(T,1,1,length(w))
        paramDesc = FD(T,1,1,1,1)
        param = Cptr[0]
        if par == 1 # matrix
            # @cudnn(cudnnGetRNNLinLayerMatrixParams,
            #       (Cptr, Cptr, Cint, #handle,rdesc, layer
            #        Cptr, Cptr, Cptr, #xDesc, wDesc, w
            #        Cint, Cptr, Ptr{Cptr}), #lid, lmatdesc, linlayermat
            #       handle, r.rnnDesc, layer-1,
            #       xDesc, wDesc, w,
            #       id-1, paramDesc, param)
            CUDNN.cudnnGetRNNLinLayerMatrixParams(handle, r.rnnDesc, layer-1, xDesc, wDesc, w, id-1, paramDesc, param)
        else # bias
            # @cudnn(cudnnGetRNNLinLayerBiasParams,
            #       (Cptr, Cptr, Cint, #handle,rdesc, layer
            #        Cptr, Cptr, Cptr, #xDesc, wDesc, w
            #        Cint, Cptr, Ptr{Cptr}), #lid, lmatdesc, linlayermat
            #       handle, r.rnnDesc, layer-1,
            #       xDesc, wDesc, w,
            #       id-1, paramDesc, param)
            CUDNN.cudnnGetRNNLinLayerBiasParams(handle, r.rnnDesc, layer-1, xDesc, wDesc, w, id-1, paramDesc, param)
        end
        dt,sz = cudnnGetFilterNdDescriptor(paramDesc)
        if should_return_nothing
            @assert param[1] === C_NULL
            @assert sz == ()
            return nothing
        end
        len = prod(sz)
        i1 = 1 + div(Int(param[1] - pointer(w)), sizeof(T))
        i2 = i1 + len - 1
    else # if isa(w, KnetArray)
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
    if i1 > i2
        @assert should_return_nothing
        nothing
    elseif par == 1 # matrix; weights are transposed
        h = Int(r.hiddenSize)
        reshape(access(r.w, i1:i2),:,h)
    else # bias
        access(r.w, i1:i2)
    end
end


"""
    rnnparams(r::RNN)

Return the RNN parameters as an Array{Any}.

The order of params returned (subject to change):
* All weight matrices come before all bias vectors.
* Matrices and biases are sorted lexically based on (layer,id).
* See @doc rnnparam for valid layer and id values.
* Input multiplying matrices are `nothing` if r.inputMode = 1.
"""
function rnnparams(r::RNN; handle=gethandle(), useview=false)
    layers = r.numLayers * (r.direction == 1 ? 2 : 1)
    ids = rnnids(r)
    ws = []
    for m in (1,2)
        for l in 1:layers
            for i in 1:ids
                push!(ws, rnnparam(r, l, i, m; handle=handle, useview=useview))
            end
        end
    end
    return ws
end

function rnnforw(r::RNN, w::KnetArray{T,3}, x::KnetArray{T},
                 hx::Union{KnetArray{T},Nothing}=nothing,
                 cx::Union{KnetArray{T},Nothing}=nothing;
                 handle=CUDNN.handle(),
                 batchSizes=nothing,
                 hy = (hx != nothing),
                 cy = (cx != nothing && r.mode == 2),
                 ) where {T}
    @assert w === value(r.w)
    # Input descriptors
    if size(x,1) != r.inputSize
        throw(DimensionMismatch("size(x,1)=$(size(x,1)) does not match r.inputSize=$(r.inputSize)"))
    end
    seqLength = batchSizes==nothing ? size(x,3) : length(batchSizes) # (X,B,T) or (X,B+) with batchSizes
    wDesc = FD3(w)              # (1,1,W)
    xtds = TDs(x,batchSizes)    # (1,X,Bt) x T
    isnothing(a) = a === nothing || a === C_NULL || a === CU_NULL
    if hx==nothing; hx=CU_NULL; hxDesc=C_NULL; else; hxDesc=TD3(hx); end # (H,B,L/2L)
    if cx==nothing || r.mode != 2; cx=CU_NULL; cxDesc=C_NULL; else; cxDesc=TD3(cx); end

    # Output arrays and descriptors
    ysize = collect(size(x))
    ysize[1] = r.hiddenSize * (r.direction == 1 ? 2 : 1)
    y = similar(x, ysize...)    # (H/2H,B,T) or (H/2H,B+) -- y mirrors x except for the first dimension
    ytds = TDs(y,batchSizes)    # (1,H/2H,Bt) x T

    # Optionally output hidden and cell of last step
    hyout = cyout = CU_NULL
    hyDesc = cyDesc = C_NULL
    if hy || cy
        firstBatchSize = batchSizes==nothing ? size(x,2) : batchSizes[1]
        hsize = (Int(r.hiddenSize), Int(firstBatchSize), Int(r.numLayers * (r.direction == 1 ? 2 : 1))) # (H,B,L/2L)
        if hy; hyout=similar(y,hsize); hyDesc=TD3(hyout); end
        if cy && r.mode==2; cyout=similar(y,hsize); cyDesc=TD3(cyout); end
        if !isnothing(hx) && any(size(hx,i)!=hsize[i] for i=1:3) # compare one by one in case hx is 1-D or 2-D
            throw(DimensionMismatch("size(hx)=$(size(hx)) does not match hsize=$(hsize)"))
        end
        if !isnothing(cx) && r.mode == 2 && any(size(cx,i)!=hsize[i] for i=1:3)
            throw(DimensionMismatch("size(cx)=$(size(cx)) does not match hsize=$(hsize)"))
        end
    end

    # workSpace and reserveSpace
    wss = cudnnGetRNNWorkspaceSize(r.rnnDesc, xtds; handle=handle)
    ws = KnetArray{UInt8}(undef,wss) # cudnnWorkSpace(wss)

    if training()
        rss = cudnnGetRNNTrainingReserveSize(r.rnnDesc, xtds; handle=handle)
        rs = KnetArray{UInt8}(undef,rss)
        # @cudnn(cudnnRNNForwardTraining,
        #       (Cptr, Cptr, Cint,  # handle,rnnDesc,seqLength
        #        Ptr{Cptr}, Ptr{T}, #x
        #        Cptr, Ptr{T}, #hx
        #        Cptr, Ptr{T}, #cx
        #        Cptr, Ptr{T}, #w
        #        Ptr{Cptr}, Ptr{T}, #y
        #        Cptr, Ptr{T}, #hy
        #        Cptr, Ptr{T}, #cy
        #        Cptr, Csize_t, #ws
        #        Cptr ,Csize_t#rs
        #        ),
        #       handle, r.rnnDesc, seqLength,
        #       xtds, x,
        #       hxDesc, hx,
        #       cxDesc, cx,
        #       wDesc, w,
        #       ytds, y,
        #       hyDesc, hyout,
        #       cyDesc, cyout,
        #       ws, wss,
        #       rs, rss)
        CUDNN.cudnnRNNForwardTraining(handle, r.rnnDesc, seqLength, xtds, x, hxDesc, hx, cxDesc, cx, wDesc, w, ytds, y, hyDesc, hyout, cyDesc, cyout, ws, wss, rs, rss)
    else
        rs = nothing
        # @cudnn(cudnnRNNForwardInference,
        #       (Cptr, Cptr, Cint,  # handle,rnnDesc,seqLength
        #        Ptr{Cptr}, Ptr{T}, #x
        #        Cptr, Ptr{T}, #h
        #        Cptr, Ptr{T}, #c
        #        Cptr, Ptr{T}, #w
        #        Ptr{Cptr}, Ptr{T}, #y
        #        Cptr, Ptr{T}, #hy
        #        Cptr, Ptr{T}, #cy
        #        Cptr, Csize_t, #ws
        #        ),
        #       handle, r.rnnDesc, seqLength,
        #       xtds, x,
        #       hxDesc, hx,
        #       cxDesc, cx,
        #       wDesc, w,
        #       ytds, y,
        #       hyDesc, hyout,
        #       cyDesc, cyout,
        #       ws, wss)
        CUDNN.cudnnRNNForwardInference(handle, r.rnnDesc, seqLength, xtds, x, hxDesc, hx, cxDesc, cx, wDesc, w, ytds, y, hyDesc, hyout, cyDesc, cyout, ws, wss)
    end
    if hyout === CU_NULL; hyout = nothing; end
    if cyout === CU_NULL; cyout = nothing; end
    return y, hyout, cyout, rs, ws
end

@primitive rnnforw(r::RNN, w::KnetArray, x...; o...),dy,y nothing rnnback2(dy,y,r,w,x...;o...) value(r).dx value(r).dhx value(r).dcx

function rnnback2(dt, t, r, w, x, hx=nothing, cx=nothing; o...)
    @assert r.w === w
    y,hy,cy,rs,ws = value(t)
    dy,dhy,dcy,drs,dws = value(dt)
    r=value(r); w=value(w); x=value(x); hx=value(hx); cx=value(cx)
    # To prevent dependencies to next iteration we need to clear the Result type from r.h,r.c
    # We can't do this during forward, because another forward may be run within the same iteration.
    # Doing it here is safe, means the iteration is done and we are taking gradients.
    # Note that this does not work on the cpu and these have to be cleaned by hand.
    # The cpu version is not a primitive and has no back function. (TODO: find better solution)
    r.h = value(r.h); r.c = value(r.c) 
    rnnback(r, w, x, y, dy, hx, cx, dhy, dcy, rs, ws; o...)
end

function rnnback(r::RNN, w::KnetArray{T}, x::KnetArray{T}, y::KnetArray{T},
                 dy, hx, cx, dhy, dcy, rs, ws; handle=CUDNN.handle(), batchSizes=nothing, o...) where {T}
    @assert value(r.w) === w
    # Input descriptors:
    seqLength = batchSizes==nothing ? size(x,3) : length(batchSizes) # (X,B,T) or (X,B+) with batchSizes
    wDesc = FD3(w)              # (1,1,W)
    xtds = TDs(x,batchSizes)    # (X,B,T) -> (1,X,B) x T
    ytds = TDs(y,batchSizes)    # (H/2H,B,T) -> (1,H/2H,B) x T
    # dytds = TDs(dy,batchSizes)  # we use ytds for dytds
    if dy == nothing; dy=zero(y); end
    if hx == nothing; hx=CU_NULL; hxDesc=C_NULL; else; hxDesc=TD3(hx); end
    if cx == nothing || r.mode != 2; cx=CU_NULL; cxDesc=C_NULL; else; cxDesc=TD3(cx); end
    if dhy == nothing; dhy=CU_NULL; dhyDesc=C_NULL; else; dhyDesc=TD3(dhy); end
    if dcy == nothing || r.mode != 2; dcy=CU_NULL; dcyDesc=C_NULL; else; dcyDesc=TD3(dcy); end

    # Output arrays and descriptors:
    dx = similar(x)             # (X,B,T) or (X,B+) with batchSizes
    # dxtds = TDs(dx,batchSizes)  # we use xtds here
    dw = zero(w)               # dw is used additively, so we need zero
    dwDesc = FD3(dw)
    if hx === CU_NULL; dhx=CU_NULL; dhxDesc=C_NULL; else; dhx=similar(hx); dhxDesc=TD3(dhx); end
    if cx === CU_NULL; dcx=CU_NULL; dcxDesc=C_NULL; else; dcx=similar(cx); dcxDesc=TD3(dcx); end

    # workSpace and reserveSpace
    # ws = cudnnWorkSpace()
    wss = bytes(ws)
    rss = bytes(rs)

    # data backward
    # @cudnn(cudnnRNNBackwardData,
    #       (Cptr, Cptr, Cint,  # handle, rnnDesc, seqLength
    #        Ptr{Cptr}, Ptr{T}, #y
    #        Ptr{Cptr}, Ptr{T}, #dy
    #        Cptr, Ptr{T}, #dhy
    #        Cptr, Ptr{T}, #dcy
    #        Cptr, Ptr{T}, #w
    #        Cptr, Ptr{T}, #hx
    #        Cptr, Ptr{T}, #cx
    #        Ptr{Cptr}, Ptr{T}, #dx
    #        Cptr, Ptr{T}, #dhx
    #        Cptr, Ptr{T}, #dcx
    #        Cptr, Csize_t, #ws
    #        Cptr, Csize_t), #rs
    #       # Use rtd with nullables
    #       handle, r.rnnDesc, seqLength,
    #       ytds, y,
    #       ytds, dy,
    #       dhyDesc, dhy,
    #       dcyDesc, dcy,
    #       wDesc, w,
    #       hxDesc, hx,
    #       cxDesc, cx,
    #       xtds, dx,
    #       dhxDesc, dhx,
    #       dcxDesc, dcx,
    #       ws, wss,
    #       rs, rss)
    CUDNN.cudnnRNNBackwardData(handle, r.rnnDesc, seqLength, ytds, y, ytds, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, xtds, dx, dhxDesc, dhx, dcxDesc, dcx, ws, wss, rs, rss)

    # weights backward
    # @cudnn(cudnnRNNBackwardWeights,
    #       (Cptr, Cptr, Cint,  # handle, rnnDesc, seqLength
    #        Ptr{Cptr}, Ptr{T}, #x
    #        Cptr, Ptr{T}, #hx
    #        Ptr{Cptr}, Ptr{T}, #y
    #        Cptr, Csize_t, #ws
    #        Cptr, Ptr{T}, #dw
    #        Ptr{Cptr}, Csize_t), #rs
    #       handle, r.rnnDesc, seqLength,
    #       xtds, x,
    #       hxDesc, hx,
    #       ytds, y,
    #       ws, wss,
    #       dwDesc, dw,
    #       rs, rss)
    CUDNN.cudnnRNNBackwardWeights(handle, r.rnnDesc, seqLength, xtds, x, hxDesc, hx, ytds, y, ws, wss, dwDesc, dw, rs, rss)
    # Update the cache
    if dhx===CU_NULL; dhx=nothing; end
    if dcx===CU_NULL; dcx=nothing; end
    r.dx, r.dhx, r.dcx = dx, dhx, dcx
    return dw
end

# CPU version
function rnnforw(r::RNN, w::AbstractArray{T}, x::AbstractArray{T},
                 hx::Union{AbstractArray{T},Nothing}=nothing,
                 cx::Union{AbstractArray{T},Nothing}=nothing;
                 # handle=CUDNN.handle(), training=false,
                 batchSizes=nothing,
                 hy = (hx != nothing),
                 cy = (cx != nothing && r.mode == 2),
                 o...) where {T}
    @assert w === value(r.w)
    rnntest(r,w,x,hx,cx;batchSizes=batchSizes,hy=hy,cy=cy)
end

# rnnforw is an AutoGrad primitive for KnetArray, but a regular function for AbstractArray:
rnnforw(r::RNN, w::AutoGrad.Value{<:AbstractArray}, x...; o...) = rnntest(r,w,x...;o...)


# non-CUDNN cpu/gpu version
function rnntest(r::RNN, ws, x, hx=nothing, cx=nothing;
                 batchSizes=nothing,
                 hy = (hx != nothing),
                 cy = (cx != nothing && r.mode == 2),
                 o...)
    if batchSizes != nothing
        return rnntest_bs(batchSizes,r, ws, x, hx, cx; hy=hy, cy=cy, o...)
    end
    @assert value(r.w) === value(ws)
    w = rnnparams(r)
    X,B,T = (size(x,i) for i=1:3) # ndims(x) may be 1,2 or 3
    @assert X == r.inputSize
    Y = Int(r.hiddenSize * (r.direction == 1 ? 2 : 1))
    ysize = ntuple(i->(i==1 ? Y : size(x,i)), ndims(x)) # to match ndims(y) to ndims(x)
    H = Int(r.hiddenSize)
    #@assert (r.inputMode == 0 || H == X)
    L = Int(r.numLayers) * (r.direction == 1 ? 2 : 1)
    hsize = (H, B, L)
    @assert hx == nothing || eqsize(size(hx), hsize)
    @assert cx == nothing || eqsize(size(cx), hsize)
    h = hx==nothing ? fill!(similar(value(x),hsize),0) : hx
    #  hs = Array{Any}[ h[:,:,l] for l=1:L ]
    hs = Array{Any}(undef,L)
    for l = 1:L
        hs[l] = h[:,:,l]
    end
    ys = []
    direction = r.direction
    pdrop = r.dropout
    #=
    All complexity of bidirectional execution
    is packed inside this inline function.
    This causes code repetition, but  works w/o
    touching the existing unidirectional test code
    =#
    @inline bidirect(update_h!) = begin
        xl = x
        for l = 1:(1+direction):L
            skip = l==1 && r.inputMode==1
            hts = []
            if l>1; xl = dropout(xl, pdrop); end
            for t = 1:T
                for (i,ti) in zip([l, l+1], [t, T-t+1])
                    # this function updates h[i]
                    update_h!(xl, i, ti, skip)
                    push!(hts, hs[i])
                end
            end
            # construct the next layer input
            yforw = Array{Any}(hts[1:2:end-1])
            yback = Array{Any}(reverse(hts[2:2:end]))
            ybs = []
            for (yf, yb) in zip(yforw, yback)
                push!(ybs, vcat(yf, yb))
            end
            # now ybs contans (2 * hiddenSize, batchSize) matrices
            # so cat them to add time dimension
            xl = reshape(hcat(ybs...), (2r.hiddenSize, size(x,2), length(ybs)))
        end
        ys = xl
    end

    if r.mode <= 1
        #@assert r.inputMode == 0 || all(w[1:1+r.direction] .== nothing)
        f = r.mode == 0 ? relu : tanh
        if direction == 0
            for t = 1:T
                for l = 1:L
                    xl = l>1 ? dropout(hs[l-1], pdrop) : nothing
                    wx,wh,bx,bh = w[2l-1],w[2l],w[2L+2l-1],w[2L+2l]
                    wxt = (l > 1 ? wx' * xl : r.inputMode==0 ? wx' * x[:,:,t] : x[:,:,t])
                    hs[l] = f.(wxt .+ wh' * hs[l] .+ bx .+ bh)
                end
                push!(ys, hs[L])
            end
        else
            bidirect() do xl, i, ti, skip
                wx,wh,bx,bh = w[2i-1],w[2i],w[2L+2i-1],w[2L+2i]
                wxt =  skip ? xl[:,:,ti] : wx' * xl[:,:,ti]
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
        c = cx==nothing ? fill!(similar(value(x),hsize),0) : cx
        cs = Array{Any}(undef,L)
        for l = 1:L
            cs[l] = c[:,:,l]
        end
        if direction == 0
            for t = 1:T
                for l = 1:L
                    xl = l>1 ? dropout(hs[l-1], pdrop) : nothing
                    Wi,Wf,Wc,Wo,Ri,Rf,Rc,Ro = w[1+8*(l-1):8l]
                    bWi,bWf,bWc,bWo,bRi,bRf,bRc,bRo = w[8L+1+8*(l-1):8L+8l]
                    Wixt = (l > 1 ? Wi' * xl : r.inputMode==0 ? Wi' * x[:,:,t] : x[:,:,t])
                    Wfxt = (l > 1 ? Wf' * xl : r.inputMode==0 ? Wf' * x[:,:,t] : x[:,:,t])
                    Wcxt = (l > 1 ? Wc' * xl : r.inputMode==0 ? Wc' * x[:,:,t] : x[:,:,t])
                    Woxt = (l > 1 ? Wo' * xl : r.inputMode==0 ? Wo' * x[:,:,t] : x[:,:,t])
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
            bidirect() do xl, i, ti, skip
                Wi,Wf,Wc,Wo,Ri,Rf,Rc,Ro = w[1+8*(i-1):8i]
                bWi,bWf,bWc,bWo,bRi,bRf,bRc,bRo = w[8L+1+8*(i-1):8L+8i]
                Wixt = skip ? xl[:,:,ti] : Wi' * xl[:,:,ti]
                Wfxt = skip ? xl[:,:,ti] : Wf' * xl[:,:,ti]
                Wcxt = skip ? xl[:,:,ti] : Wc' * xl[:,:,ti]
                Woxt = skip ? xl[:,:,ti] : Wo' * xl[:,:,ti]
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
                    xl = l>1 ? dropout(hs[l-1], pdrop) : nothing
                    Wr,Wi,Wh,Rr,Ri,Rh = w[1+6*(l-1):6l]
                    bWr,bWi,bWh,bRr,bRi,bRh = w[6L+1+6*(l-1):6L+6l]
                    Wrxt = (l > 1 ? Wr' * xl : r.inputMode==0 ? Wr' * x[:,:,t] : x[:,:,t])
                    Wixt = (l > 1 ? Wi' * xl : r.inputMode==0 ? Wi' * x[:,:,t] : x[:,:,t])
                    Whxt = (l > 1 ? Wh' * xl : r.inputMode==0 ? Wh' * x[:,:,t] : x[:,:,t])
                    rt = sigm.(Wrxt .+ Rr' * hs[l] .+ bWr .+ bRr)
                    it = sigm.(Wixt .+ Ri' * hs[l] .+ bWi .+ bRi)
                    ht = tanh.(Whxt .+ rt .* (Rh' * hs[l] .+ bRh) .+ bWh)
                    hs[l] = (1 .- it) .* ht .+ it .* hs[l]
                end
                push!(ys, hs[L])
            end
        else
            bidirect() do xl, i, ti, skip
                Wr,Wi,Wh,Rr,Ri,Rh = w[1+6*(i-1):6i]
                bWr,bWi,bWh,bRr,bRi,bRh = w[6L+1+6*(i-1):6L+6i]
                Wrxt = skip ? xl[:, :, ti] : Wr' * xl[:, :, ti]
                Wixt = skip ? xl[:, :, ti] : Wi' * xl[:, :, ti]
                Whxt = skip ? xl[:, :, ti] : Wh' * xl[:, :, ti]
                rt = sigm.(Wrxt .+ Rr' * hs[i] .+ bWr .+ bRr)
                it = sigm.(Wixt .+ Ri' * hs[i] .+ bWi .+ bRi)
                ht = tanh.(Whxt .+ rt .* (Rh' * hs[i] .+ bRh) .+ bWh)
                hs[i] = (1 .- it) .* ht .+ it .* hs[i]
            end
        end
    else
        error("RNN not supported")
    end
    y = r.direction == 0 ? reshape(hcat(ys...), ysize) : reshape(ys,ysize)
    hyout = hy ? reshape(hcat(hs...), hsize) : nothing
    cyout = cy && r.mode == 2 ? reshape(hcat(cs...), hsize) : nothing
    return (y,hyout,cyout,nothing,nothing)
end

# compare sizes ignoring trailing ones
function eqsize(a, b)
    na = length(a)
    nb = length(b)
    (na == nb ? a == b : na > nb ? 
     a[1:nb] == b && all(a[nb+1:end] .== 1) :
     b[1:na] == a && all(b[na+1:end] .== 1))
end


# TODO: WIP
function rnntest_bs(batchSizes, r::RNN, w, x,
                    hx=nothing, cx=nothing;
                    # handle=CUDNN.handle(), training=false,
                    hy = (hx != nothing),
                    cy = (cx != nothing && r.mode == 2),
                    o...)
    # TODO: fix this implementation
    error("Implementation of batchSizes is not completed in CPU")
    @assert value(r.w) === value(w)
    # Here needs reshaping hidden sizes
    if length(Set{Int}(batchSizes)) == 1
        x_ = reshape(x, (r.inputSize, div(size(x,2), batchSizes[1]), batchSizes[1]))
        y,hy,cy,rs = rnntest(r, w, x_, hx, cx; hy=hy, cy=cy, o...)
        y = reshape(y, (size(y, 1), size(y,2) * size(y,3)))
        return y, hy, cy, rs
    end
    hrem(h, bs1, bs2) = (bs2 < bs1) ? (h[:, 1:bs1-bs2+1, l] for l=1:size(h,3)) : nothing
    hnext(h, bs1, bs2) = (bs2 < bs1) ? h[:, 1:bs2, :] : h
    hrems = []
    ys = []
    crems = []
    ind = 1
    for i = 1:length(batchSizes)
        xt = x[:, ind:ind+batchSizes[i]-1]
        xt = reshape(xt, size(xt)..., 1)
        y, hx, cx = rnntest(r, w, xt, hx, cx;
                            hy=true, cy=true)
        if i > 1
            hr = hrem(hx,batchSizes[i],batchSizes[i+1])
            if hr !== nothing
                hy && push!(hrems, hr...)
                hx = hnext(hx, batchSizes[i],batchSizes[i+1])
            end
            if r.mode == 2
                cr = crem(h,batchSizes[i], batchSizes[i+1])
                if cr !== nothing
                    cy && push!(crems, cr...)
                    cx = hnext(cx, batchSizes[i],batchSizes[i+1])
                end
            end
        end
        ind += batchSizes[i]
        push!(ys, reshape(y, size(y,1,2))) #only last layer output
    end
    # reconstruct the output
    ## hx has size (h,bs[end],l)
    ## hrems has (hs,bsi) dimensional matrices
    hout(hx, hy, hrems) = begin
        nlayers = size(hx, 3)
        if hy
            hts = []
            for i = 1:nlayers
                push!(hts, hy[:,:,i])
            end
            hyouts = []
            for l = 1:nlayers
                push!(hyouts, hcat(hts[l], reverse(hrems[l:nlayers:end-nlayers+l])...))
            end
            hsize = (size(hyouts[end])..., nlayers)
            reshape(length(hyouts) > 1 ? hcat(hyouts...) : hyouts[1], hsize)
        else
            nothing
        end
    end
    return (hcat(ys...), hout(hx, hy, hrems), r.mode==2 ? hout(cx, cy, crems) : nothing, nothing)
end


## DEPRECATED:
function rnninit(x...; o...)
    @warn "rnninit is deprecated, use RNN instead" maxlog=1
    r=RNN(x...; o...)
    return (r,r.w)
end

function rnnparams(r,w;o...)
    @warn "rnnparams(r,w) is deprecated, use rnnparams(r) instead" maxlog=1
    @assert value(w)===value(r.w)
    rnnparams(r;o...)
end

function rnnparam(r,w,l,i,d;o...)
    @warn "rnnparam(r,w,l,i,d) is deprecated, use rnnparam(r,l,i,d) instead" maxlog=1
    @assert value(w)===value(r.w)
    rnnparam(r,l,i,d;o...)
end


## #506: Because r.dx,dhx,dcx may be freed by gcnode, their C_NULL pointers cause trouble in deepcopy.
import Base: deepcopy_internal
function deepcopy_internal(x::RNN, s::IdDict)
    if !haskey(s,x)
        s[x] = RNN(deepcopy_internal(x.w,s), deepcopy_internal(x.h,s), deepcopy_internal(x.c,s), x.inputSize, x.hiddenSize, x.numLayers, x.dropout, x.seed, x.inputMode, x.direction, x.mode, x.algo, x.dataType, deepcopy_internal(x.rnnDesc,s), deepcopy_internal(x.dropoutDesc,s), nothing, nothing, nothing)
    end
    return s[x]
end
