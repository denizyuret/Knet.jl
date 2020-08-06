export RNN, rnninit, rnnforw, rnnparam, rnnparams # TODO: we shouldn't export structs like RNN from ops
using Knet: atype, training # TODO: ops should not initialize params
using AutoGrad: Param, value # TODO: ops should not use Param
import Base: show

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
- `algo=0`: Algorithm to use, see CUDNN docs for details.
- `seed=0`: Random number seed for dropout. Uses `time()` if 0.
- `winit=xavier`: Weight initialization method for matrices.
- `binit=zeros`: Weight initialization method for bias vectors.
- `finit=ones`: Weight initialization method for the bias of forget gates.
- `atype=Knet.atype()`: array type for model weights.

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
    rnnDesc
    dropoutDesc
    dx
    dhx
    dcx
end

function RNN(inputSize, hiddenSize; h=nothing, c=nothing,
             rnnType=:lstm,       # CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3
             bidirectional=false, # CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL = 1
             skipInput=false,     # CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT = 1
             numLayers=1,
             dropout=0.0,
             winit=xavier,
             binit=zeros,
             finit=ones,          # forget bias for lstm
             algo=0,              # CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
             seed=0,              # seed=0 for random init, positive integer for replicability
             atype=atype(),
             # deprecated
             dataType=nothing,    # CUDNN_DATA_FLOAT  = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_DATA_HALF   = 2
             usegpu=nothing,
             handle=nothing,
             )
    if handle !== nothing
        @warn "The handle option is deprecated, GPU implementation using CUDNN.handle()" maxlog=1
    end
    if dataType !== nothing || usegpu !== nothing
        @warn "dataType and usegpu options are deprecated, using atype=$atype" maxlog=1
    end
    dataType = eltype(atype)
    usegpu = !(atype <: Array)

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
    # These should be set by the gpu implementations during first rnnforw:
    dropoutDesc = nothing # usegpu ? DD(handle=handle,dropout=dropout,seed=seed) : nothing # Need to keep dropoutDesc in RNN so it does not get gc'ed.
    rnnDesc = nothing # usegpu ? RD(hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,dataType) : nothing
    r = RNN(w,h,c,inputSize,hiddenSize,numLayers,dropout,seed,inputMode,direction,mode,algo,dataType,rnnDesc,dropoutDesc,dx,dhx,dcx)

    r.w = Array{dataType}(undef,1,1,getRNNParamsSize(r))
    for a in rnnparams(r; useview=true)
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
            a = rnnparam(r, layer, id, param, useview=true)
            if a != nothing
                copyto!(a, finit(dataType, size(a)))
            end
        end
    end
    # many copyto! ops to gpu is expensive (~20s), so we init on cpu and copy it over once here.
    r.w = convert(atype, r.w)
    r.w = Param(r.w)
    return r
end

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
function rnnparam(r::RNN, layer::Integer, id::Integer, par::Integer; useview=true)
    params_are_good = 
    ((1 <= par <= 2) &&
     ((r.direction == 0 && 1 <= layer <= r.numLayers) ||
      (r.direction == 1 && 1 <= layer <= 2*r.numLayers)) &&
     ((r.mode == 0 && 1 <= id <= 2) ||
      (r.mode == 1 && 1 <= id <= 2) ||
      (r.mode == 2 && 1 <= id <= 8) ||
      (r.mode == 3 && 1 <= id <= 6)))
    params_are_good || throw(ArgumentError("Bad arguments for rnnparam, please see `@doc rnnparam`."))
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
    if i1 > i2
        @assert should_return_nothing
        nothing
    elseif par == 1 # matrix; weights are transposed
        h = Int(r.hiddenSize)
        reshape(view(r.w, i1:i2),:,h)
    else # bias
        view(r.w, i1:i2)
    end
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

"""
    rnnparams(r::RNN)

Return the RNN parameters as an Array{Any}.

The order of params returned (subject to change):
* All weight matrices come before all bias vectors.
* Matrices and biases are sorted lexically based on (layer,id).
* See @doc rnnparam for valid layer and id values.
* Input multiplying matrices are `nothing` if r.inputMode = 1.
"""
function rnnparams(r::RNN; useview=false)
    layers = r.numLayers * (r.direction == 1 ? 2 : 1)
    ids = rnnids(r)
    ws = []
    for m in (1,2)
        for l in 1:layers
            for i in 1:ids
                push!(ws, rnnparam(r, l, i, m; useview=useview))
            end
        end
    end
    return ws
end

# TODO: instead of rnnparam, rnnparams, have the views represented inside the struct.

rnnids(r) = (r.mode == 2 ? 8 : r.mode == 3 ? 6 : 2)

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
    # TODO: do this check on unit testing
    # if r.rnnDesc != nothing
    #     @assert nparams == cudnnGetRNNParamsSize(r)
    # end
    return nparams
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
    # TODO: the cpu implementation does not respect the seed parameter.
    # TODO: reconsider dropout for the input in next release.
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


# CPU version: need to keep the name rnntest for unit testing
rnnforw(r::RNN, x...; o...) = rnntest(r, x...; o...)

# TODO: interface consistency, deprecate all r,w signatures, have r, signatures. Right now
# rnninit, rnnparam, rnnparams have r,w deprecated. rnnforw is a more difficult case because
# of the optional hx, cx parameters.

function rnntest(r::RNN, ws, x, hx=nothing, cx=nothing;
                 batchSizes=nothing,
                 hy = (hx != nothing),
                 cy = (cx != nothing && r.mode == 2))
    @assert batchSizes == nothing "batchSizes option has not been implemented for the CPU yet."
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

## TO BE DEPRECATED:
function rnninit(x...; o...)
    # @warn "rnninit is deprecated, use RNN instead" maxlog=1
    r=RNN(x...; o...)
    return (r,r.w)
end

function rnnparams(r,w;o...)
    # @warn "rnnparams(r,w) is deprecated, use rnnparams(r) instead" maxlog=1
    @assert value(w)===value(r.w)
    rnnparams(r;o...)
end

function rnnparam(r,w,l,i,d;o...)
    # @warn "rnnparam(r,w,l,i,d) is deprecated, use rnnparam(r,l,i,d) instead" maxlog=1
    @assert value(w)===value(r.w)
    rnnparam(r,l,i,d;o...)
end

function xavier(a...; gain=1)
    w = rand(a...)
    @assert ndims(w) == 2
    fanout, fanin = size(w)
    s = convert(eltype(w), gain*sqrt(6 / (fanin + fanout)))
    return 2s .* w .- s
end
