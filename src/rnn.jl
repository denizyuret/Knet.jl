using Knet

Knet.cudnnhandle()
using Knet: @cuda, cudnnhandle, Cptr, cudnnVersion, bytes, FD, DT, TD
using AutoGrad: Rec, Grad, recorder
import Knet.DT

type DD; ptr; states; end       # TODO: Can multiple RNNs share dropout descriptors? Can dropout probability be changed?
function DD(; handle=cudnnhandle(), dropout=0.0, seed=42, o...)
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

type RD
    ptr
    dropoutDesc
    # store metadata here
    hiddenSize
    numLayers
    mode
    dataType
    inputMode
    direction
    # add stuff as needed
end

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
            seed=42,
            o...
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
    rd = RD(d[1], dropoutDesc, hiddenSize, numLayers, mode, dataType,inputMode,direction)
    finalizer(rd, x->@cuda(cudnn,cudnnDestroyRNNDescriptor,(Cptr,),x.ptr))
    return rd
end

DT(::Type{Float32}) = 0
DT(::Type{Float64}) = 1
DT(::Type{Float16}) = 2
import Base.unsafe_convert
unsafe_convert(::Type{Cptr}, td::RD)=td.ptr
unsafe_convert(::Type{Cptr}, td::DD)=td.ptr

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

function workspace_size(rd::RD, tds;
                        handle=cudnnhandle(),
                        o...)
    seqlength = length(tds)
    res = Csize_t[1]
    tds = Cptr[td.ptr for td in tds]
    @cuda(cudnn, cudnnGetRNNWorkspaceSize,
          # handle, rnndesc, seqlength, xdesc, res        ,
          (Cptr,  Cptr, Cint, Ptr{Cptr}, Ptr{Csize_t}),
          handle, rd.ptr, seqlength, tds, res)
    return Int(res[1])
end

function reserved_size(rd::RD, tds,
                       handle=cudnnhandle(),o...)
    seqlength = length(tds)
    res = Csize_t[1]
    tds = Cptr[td.ptr for td in tds]
    @cuda(cudnn, cudnnGetRNNTrainingReserveSize,
          # handle, rnndesc, seqlength, xdesc, res        ,
          (Cptr,  Cptr, Cint, Ptr{Cptr}, Ptr{Csize_t}),
          handle, rd.ptr, seqlength, tds, res)
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
