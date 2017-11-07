using Knet

#=
LOW LEVEL API USER WON'T ACCESS
=#
# TODO: remove this
Knet.cudnnhandle() #without calling this, cudnnVersion is not generated
using Knet: @cuda, cudnnhandle,Cptr, cudnnVersion
#bytes(a) = length(a) * sizeof(eltype(a))
using AutoGrad: Rec, Grad, recorder
include("cudnn.jl")

function workspace_size(rd::RD, tds::Array{TD, 1};
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

function reserved_size(rd::RD, tds::Array{TD, 1},
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
    if xtdims==nothing; xtdims = _xtdims(rc.inputSize); end #we want this to work
    eltype = JDT(rc.rd.dataType)
    res = Csize_t[1]
    #tds = Cptr[TD(xtdims).ptr]
    @cuda(cudnn, cudnnGetRNNParamsSize,
          # handle, rnndesc, seqlength, xdesc, res
          (Cptr,  Cptr, Cptr, Ptr{Csize_t}, UInt32),
          handle, rc.rd, TD(xtdims, rc.rd.dataType), res, rc.rd.dataType)
    return div(Int(res[1]), sizeof(eltype))
end

function init_params(rc::RNNCache; handle=cudnnhandle(),o...)
    eltype = JDT(rc.rd.dataType)
    params = KnetArray{eltype}(1,1,nparams(rc; handle=handle,o...))
    return params
end

# Parameter collection and re-collection stuff
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


# this function should be users only 
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
    xtds = [TD(xtdims, cT) for i=1:seqlength]
    # allocate the workspace
    wss = workspace_size(cache.rd, xtds;o...)
    ws = getws(wss; o...)
    # allocata the reserved spave
    # TODO: can we do better in terms of memory?
    rss = reserved_size(cache.rd, xtds;o...)
    rs = training ? nothing : KnetArray{Int8}(rss)
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
    ytds = [TD(hidden_size, cT) for i=1:seqlength]
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
              TD(hidden_size, cT), hx,
              TD(hidden_size, cT), cx,
              FD(w), w,
              ytds, y,
              TD(hidden_size, cT), hy,
              TD(hidden_size, cT), cy,
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
              TD(hidden_size, cT), hx,
              TD(hidden_size, cT), cx,
              FD(w), w,
              ytds, y,
              TD(hidden_size, cT), hy,
              TD(hidden_size, cT), cy,
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
    xtds = [TD(xtdims, cT).ptr for i=1:seqlength]
    ytds = [TD(hidden_size, cT).ptr for i=1:seqlength]
    dytds = [TD(hidden_size, cT).ptr for i=1:seqlength]
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
          handle, cache.rd, seqlength,
          ytds, y,
          dytds, dy,
          TD(hidden_size, cT), dhy,
          TD(hidden_size, cT), dcy,
          FD(w), w,
          TD(hidden_size, cT), hx,
          TD(hidden_size, cT), cx,
          TD(size(dx), cT), dx,
          TD(size(dhx), cT), dhx,
          TD(size(dcx), cT), dcx,
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
          TD(hidden_size, cT), hx,
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
    # figure out a to do this
    rnn(::Type{Grad{1}},
        dy, dhy, dcy, y, w, x,
        hx, cx, hy, cy, cache, rs; o...) =
            rnn_backw(
                dy, dhy, dcy, y, w, x,
                hx, cx, hy, cy, cache, rs;
                o...)
    rnn(::Type{Grad{2}},dr,r,w,x,hx,cx,cache) = cache.dx
    rnn(::Type{Grad{3}},dr,r,w,x,hx,cx,cache) = cache.dhx
    rnn(::Type{Grad{3}},dr,r,w,x,hx,cx,cache) = cache.dcx
end
