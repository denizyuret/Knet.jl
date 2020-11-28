import Knet.Ops20: rnnforw
using Knet.Ops20: RNN
using Knet.KnetArrays: DevArray, KnetArray, Cptr
using CUDA: CuArray, CUDNN, CU_NULL
using AutoGrad: AutoGrad, @primitive1, value, recording, Param, Value

"RNN descriptor"
mutable struct RD; ptr; end

"Dropout descriptor"
mutable struct DD; ptr; states; end

"Keeps an array of 3D tensor descriptors"
mutable struct TDs; pvec::Vector{Cptr}; xDesc::Vector{TD}; end     # Keep xDesc in TDs so it does not get gc'ed

Base.unsafe_convert(::Type{Cptr}, dd::DD)=dd.ptr
Base.unsafe_convert(::Type{Cptr}, rd::RD)=rd.ptr
Base.unsafe_convert(::Type{Ptr{Cptr}}, tds::TDs)=pointer(tds.pvec)

function DD(; atype, handle=CUDNN.handle(), dropout=0.0, seed=0, o...)
    if seed==0; seed=floor(Culonglong,time()); end
    d = Cptr[0]; s = Csize_t[0] # TODO: Can multiple RNNs share dropout descriptors? Can dropout probability be changed?
    CUDNN.cudnnCreateDropoutDescriptor(d)
    CUDNN.cudnnDropoutGetStatesSize(handle,s)
    states = rnnworkspace(s[1], atype)
    @cudnn_retry CUDNN.unsafe_cudnnSetDropoutDescriptor(d[1],handle,dropout,states,bytes(states),seed)
    dd = DD(d[1],states)
    finalizer(x->CUDNN.cudnnDestroyDropoutDescriptor(x.ptr),dd)
    return dd
end

function RD()
    d = Cptr[0]
    @cudnn_retry CUDNN.unsafe_cudnnCreateRNNDescriptor(d)
    rd = RD(d[1])
    finalizer(x->CUDNN.cudnnDestroyRNNDescriptor(x.ptr),rd)
    return rd
end

function RD(hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,dataType; handle=CUDNN.handle())
    rnnDesc = RD()
    inputMode = CUDNN.cudnnRNNInputMode_t(inputMode)
    direction = CUDNN.cudnnDirectionMode_t(direction)
    mode = CUDNN.cudnnRNNMode_t(mode)
    algo = CUDNN.cudnnRNNAlgo_t(algo)
    dt = CUDNN.cudnnDataType_t(DT(dataType))
    if CUDNN.version() < v"8"
        CUDNN.cudnnSetRNNDescriptor(handle,rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,dt)
    else
        CUDNN.cudnnSetRNNDescriptor_v6(handle,rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,dt)
    end
    return rnnDesc
end

Base.length(tds::TDs)=length(tds.pvec)

function TDs(x::DevArray{A},::Nothing) where {A} # Treat x: (X,B?,T?) as a 4D array: (1,X,B,T)
    xDesc = TD(A,1,size(x,1),size(x,2)) # we can use a single xDesc
    pvec = Vector{Cptr}(undef, size(x,3))
    pvec[:] .= xDesc.ptr
    return TDs(pvec, [xDesc])
end

function TDs(x::DevArray{A},batchSizes) where {A} # x: (X,B*), batchSizes gives us Bt sizes
    @assert sum(batchSizes) == div(length(x),size(x,1))
    X = size(x,1)
    xs = [ TD(A,1,X,B) for B in batchSizes ]
    ps = [ xd.ptr for xd in xs ]
    return TDs(ps,xs)
end

function TD3(a::DevArray) # Treat a as a 3D array, pad from right
    n = ndims(a)
    if n==3; TD(a)
    elseif n==2; TD(reshape(a, size(a,1), size(a,2), 1))
    elseif n==1; TD(reshape(a, size(a,1), 1, 1))
    else; throw(DimensionMismatch())
    end
end

function FD3(a::DevArray) # Treat a as a 3D array, pad from left
    n = ndims(a)
    if n==3; FD(a)
    elseif n==2; FD(reshape(a, 1, size(a,1), size(a,2)))
    elseif n==1; FD(reshape(a, 1, 1, size(a,1)))
    else; throw(DimensionMismatch())
    end
end


function rnnforw(r::RNN, w, x::Union{DevArray{T},Value{<:DevArray{T}}}, hx=nothing, cx=nothing;
                 handle=CUDNN.handle(), batchSizes=nothing, hy = (hx != nothing), cy = (cx != nothing && r.mode == 2)) where T
    @assert value(w) === value(r.w)
    @assert size(x,1) == r.inputSize
    x3 = reshape(value(x), size(x,1), size(x,2), size(x,3))
    @assert typeof(x3) == typeof(value(w)) "$(typeof(value(w))) weights do not match $(typeof(x)) input. Please use RNN(;atype) option."
    if r.rnnDesc === nothing    # initialize rnn for gpu with first input
        r.dataType = eltype(x3)
        r.dropoutDesc = DD(handle=CUDNN.handle(),dropout=r.dropout,seed=r.seed,atype=typeof(x3))
        r.rnnDesc = RD(r.hiddenSize,r.numLayers,r.dropoutDesc,r.inputMode,r.direction,r.mode,r.algo,r.dataType)
    end
    _rnnforw(w,x,hx,cx; rnn=r,handle=handle,batchSizes=batchSizes,hy=hy,cy=cy)
end


function _rnnforw(w, x, hx, cx; rnn, handle, batchSizes, hy, cy)
    # Input descriptors
    seqLength = batchSizes==nothing ? size(x,3) : length(batchSizes) # (X,B,T) or (X,B+) with batchSizes
    wDesc = FD3(w)              # (1,1,W)
    xtds = TDs(x,batchSizes)    # (1,X,Bt) x T
    isnothing(a) = a === nothing || a === C_NULL || a === CU_NULL
    if hx==nothing; hx=CU_NULL; hxDesc=C_NULL; else; hxDesc=TD3(hx); end # (H,B,L/2L)
    if cx==nothing || rnn.mode != 2; cx=CU_NULL; cxDesc=C_NULL; else; cxDesc=TD3(cx); end

    # Output arrays and descriptors
    ysize = collect(size(x))
    ysize[1] = rnn.hiddenSize * (rnn.direction == 1 ? 2 : 1)
    y = similar(x, ysize...)    # (H/2H,B,T) or (H/2H,B+) -- y mirrors x except for the first dimension
    ytds = TDs(y,batchSizes)    # (1,H/2H,Bt) x T

    # Optionally output hidden and cell of last step
    hyout = cyout = CU_NULL
    hyDesc = cyDesc = C_NULL
    if hy || cy
        firstBatchSize = batchSizes==nothing ? size(x,2) : batchSizes[1]
        hsize = (Int(rnn.hiddenSize), Int(firstBatchSize), Int(rnn.numLayers * (rnn.direction == 1 ? 2 : 1))) # (H,B,L/2L)
        if hy; hyout=similar(y,hsize); hyDesc=TD3(hyout); end
        if cy && rnn.mode==2; cyout=similar(y,hsize); cyDesc=TD3(cyout); end
        if !isnothing(hx) && any(size(hx,i)!=hsize[i] for i=1:3) # compare one by one in case hx is 1-D or 2-D
            throw(DimensionMismatch("size(hx)=$(size(hx)) does not match hsize=$(hsize)"))
        end
        if !isnothing(cx) && rnn.mode == 2 && any(size(cx,i)!=hsize[i] for i=1:3)
            throw(DimensionMismatch("size(cx)=$(size(cx)) does not match hsize=$(hsize)"))
        end
    end

    # workSpace and reserveSpace
    wss = cudnnGetRNNWorkspaceSize(rnn.rnnDesc, xtds; handle=handle)
    ws = rnnworkspace(wss, typeof(value(w)))

    if AutoGrad.recording()
        rss = cudnnGetRNNTrainingReserveSize(rnn.rnnDesc, xtds; handle=handle)
        rs = rnnworkspace(rss, typeof(value(w)))
        @cudnn_retry CUDNN.unsafe_cudnnRNNForwardTraining(handle, rnn.rnnDesc, seqLength, xtds, x, hxDesc, hx, cxDesc, cx, wDesc, w, ytds, y, hyDesc, hyout, cyDesc, cyout, ws, wss, rs, rss)
    else
        rs = nothing
        @cudnn_retry CUDNN.unsafe_cudnnRNNForwardInference(handle, rnn.rnnDesc, seqLength, xtds, x, hxDesc, hx, cxDesc, cx, wDesc, w, ytds, y, hyDesc, hyout, cyDesc, cyout, ws, wss)
    end
    if hyout === CU_NULL; hyout = nothing; end
    if cyout === CU_NULL; cyout = nothing; end
    return y, hyout, cyout, rs, ws
end

function _rnnback(dt, t, w, x, hx, cx; rnn, o...)
    @assert value(rnn.w) === value(w)
    y,hy,cy,rs,ws = value(t)
    dy,dhy,dcy,drs,dws = value(dt)
    rnn=value(rnn); w=value(w); x=value(x); hx=value(hx); cx=value(cx)
    # To prevent dependencies to next iteration we need to clear the Result type from rnn.h,rnn.c
    # We can't do this during forward, because another forward may be run within the same iteration.
    # Doing it here is safe, means the iteration is done and we are taking gradients.
    # Note that this does not work on the cpu and these have to be cleaned by hand.
    # The cpu version is not a primitive and has no back function. (TODO: find better solution)
    rnn.h = value(rnn.h); rnn.c = value(rnn.c) 
    _rnnback2(rnn, w, x, y, dy, hx, cx, dhy, dcy, rs, ws; o...)
end
        
function _rnnback2(r, w, x, y, dy, hx, cx, dhy, dcy, rs, ws;
                   handle=CUDNN.handle(), batchSizes=nothing, o...) 
    @assert value(r.w) === value(w)
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
    @cudnn_retry CUDNN.unsafe_cudnnRNNBackwardData(handle, r.rnnDesc, seqLength, ytds, y, ytds, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, xtds, dx, dhxDesc, dhx, dcxDesc, dcx, ws, wss, rs, rss)
    @cudnn_retry CUDNN.unsafe_cudnnRNNBackwardWeights(handle, r.rnnDesc, seqLength, xtds, x, hxDesc, hx, ytds, y, ws, wss, dwDesc, dw, rs, rss)
    # Update the cache
    if dhx===CU_NULL; dhx=nothing; end
    if dcx===CU_NULL; dcx=nothing; end
    r.dx, r.dhx, r.dcx = dx, dhx, dcx
    return dw
end

@primitive1 _rnnforw(w,x,hx,cx; rnn, o...),dy,y  _rnnback(dy,y,w,x,hx,cx; rnn=rnn, o...) value(rnn).dx value(rnn).dhx value(rnn).dcx

#506: Because r.dx,dhx,dcx may be freed by gcnode, their C_NULL pointers cause trouble in deepcopy.
import Base: deepcopy_internal
function deepcopy_internal(x::RNN, s::IdDict)
    if !haskey(s,x)
        s[x] = RNN(deepcopy_internal(x.w,s), deepcopy_internal(x.h,s), deepcopy_internal(x.c,s), x.inputSize, x.hiddenSize, x.numLayers, x.dropout, x.seed, x.inputMode, x.direction, x.mode, x.algo, x.dataType, deepcopy_internal(x.rnnDesc,s), deepcopy_internal(x.dropoutDesc,s), nothing, nothing, nothing)
    end
    return s[x]
end

function rnnworkspace(n, type)
    n8 = (n-1)÷sizeof(Int)+1
    if type <: KnetArray
        buf = KnetArray{Int}(undef, n8)
    elseif type <: CuArray
        buf = CuArray{Int}(undef, n8)
    else
        error("$type not a known GPU array type.")
    end
    return buf
end

function cudnnGetRNNParamsSize(r::RNN)
    res = Csize_t[0]
    xDesc = TD(r.dataType, 1, r.inputSize, 1)    # xDesc: (1,X,B) where X = inputSize, B is ignored, so assume 1
    dt = CUDNN.cudnnDataType_t(DT(r.dataType))
    CUDNN.cudnnGetRNNParamsSize(CUDNN.handle(), r.rnnDesc, xDesc, res, dt)
    div(res[1], sizeof(r.dataType))
end

# This is buggy, why?
# X,H,L,I = r.inputSize, r.hiddenSize, r.numLayers, rnnids(r)
# biases = L*I
# inputMatrices = (r.inputMode == 1 ? 0 : r.direction == 1 ? I : div(I,2))
# hiddenMatrices = (r.direction == 1 ? (L-1)*I : (L-1)*I + div(I,2))
# biases * H + inputMatrices * X * H + hiddenMatrices * H * H

function cudnnGetRNNWorkspaceSize(rd::RD, tds::TDs; handle=CUDNN.handle())
    res = Csize_t[1]
    CUDNN.cudnnGetRNNWorkspaceSize(handle, rd, length(tds), tds, res)
    return Int(res[1])
end

function cudnnGetRNNTrainingReserveSize(rd::RD, tds::TDs; handle=CUDNN.handle())
    res = Csize_t[1]
    CUDNN.cudnnGetRNNTrainingReserveSize(handle, rd, length(tds), tds, res)
    return Int(res[1])
end

# Return eltype,size
function cudnnGetFilterNdDescriptor(wDesc::FD; nbDimsRequested = 8)
    dataType = Cint[0]
    format = Cint[0]
    nbDims = Cint[0]
    filterDimA = Vector{Cint}(undef,nbDimsRequested)
    CUDNN.cudnnGetFilterNdDescriptor(wDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)
    if nbDims[1] > nbDimsRequested
        cudnnGetFilterNdDescriptor(wDesc::FD; nbDimsRequested = nbDims[1])
    else
        (Float32,Float64,Float16)[1+dataType[1]],
        (filterDimA[nbDims[1]:-1:1]...,)
    end
end

function cudnnGetRNNParam(r::RNN, layer::Integer, id::Integer, par::Integer; useview=false)
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
    @assert isa(w, DevArray)
    T = eltype(w)
    xDesc = TD(T,1,r.inputSize,1)
    wDesc = FD(T,1,1,length(w))
    paramDesc = FD(T,1,1,1,1)
    param = Cptr[0]
    if par == 1 # matrix
        CUDNN.cudnnGetRNNLinLayerMatrixParams(handle, r.rnnDesc, layer-1, xDesc, wDesc, w, id-1, paramDesc, param)
    else # bias
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


# CuArray specific support: should move this to cuarrays
TD(a::CuArray{T}) where {T} = TD(T, size(a))
FD(a::CuArray{T}) where {T} = FD(T, size(a))
bytes(x::CuArray{T}) where T = length(x)*sizeof(T)


# KnetArray getindex contiguous indices already returns a view.
# We need the following for rnnparam/rnntest to work:        
Base.view(A::KnetArray, I::AbstractUnitRange{Int}) = getindex(A, I)

# This supports cpucopy/gpucopy:
import Knet.KnetArrays: _ser
function _ser(x::RNN, s::IdDict, m::Val)
    if !haskey(s,x)
        # we need rd,dd only if there is a gpu, we are not in cpumode,
        # and if we are in jldmode we are loading, not saving
        # if (CUDA.functional() && m != CPUMODE && !(m == JLDMODE && x.rnnDesc != nothing))
        #     dd = DD(dropout=x.dropout,seed=x.seed)
        #     rd = RD(x.hiddenSize,x.numLayers,dd,x.inputMode,x.direction,x.mode,x.algo,x.dataType)
        # else
        #     rd = dd = nothing
        # end

        # 20200806: We no longer need to load/save rd/dd: rnnforw will construct as needed.
        rd = dd = nothing

        # dx, dhx, dcx are temporary fields used by rnnback, they do not need to be copied
        # gcnode sets dx.ptr to C_NULL which breaks serialize, best not to try
        s[x] = RNN(_ser(x.w,s,m), _ser(x.h,s,m), _ser(x.c,s,m), x.inputSize, x.hiddenSize, x.numLayers, x.dropout, x.seed, x.inputMode, x.direction, x.mode, x.algo, x.dataType, rd, dd, nothing, nothing, nothing)
    end
    return s[x]
end

import JLD2

struct JLD2RNN; w; h; c; inputSize; hiddenSize; numLayers; dropout; seed; inputMode; direction; mode; algo; dataType; end
JLD2RNN(x::RNN) = JLD2RNN(x.w, x.h, x.c, x.inputSize, x.hiddenSize, x.numLayers, x.dropout, x.seed, x.inputMode, x.direction, x.mode, x.algo, x.dataType)
RNN(x::JLD2RNN) = RNN(x.w, x.h, x.c, x.inputSize, x.hiddenSize, x.numLayers, x.dropout, x.seed, x.inputMode, x.direction, x.mode, x.algo, x.dataType, nothing, nothing, nothing, nothing, nothing)

JLD2.writeas(::Type{RNN}) = JLD2RNN
JLD2.wconvert(::Type{JLD2RNN}, x::RNN) = JLD2RNN(x)
JLD2.rconvert(::Type{RNN}, x::JLD2RNN) = RNN(x)
