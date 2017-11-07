#= This file will store the cudnn related datatypes =#



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

JDT(cudnndtype) = (Float32, Float64, Float16)[cudnndtype+1]

type TD
    ptr
    dims
    dtype
end

function TD(dims, dtype;
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
    td = TD(d[1], dims, dtype)
    finalizer(td, x->@cuda(cudnn,cudnnDestroyTensorDescriptor,(Cptr,),x.ptr))
    return td
end

function TD(a::KnetArray)
    d = Cptr[0]
    @cuda(cudnn,cudnnCreateTensorDescriptor,(Ptr{Cptr},),d)
    n = ndims(a)
    sz = [Cint(size(a,n-i+1)) for i=1:n]
    st = [Cint(stride(a,n-i+1)) for i=1:n]
    @cuda(cudnn,cudnnSetTensorNdDescriptor,
              (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint}),
              d[1], DT(a), n, sz, st)
    td = new(d[1])
    finalizer(td, x->@cuda(cudnn,cudnnDestroyTensorDescriptor,(Cptr,),x.ptr))
    return td
end

type FD
    ptr
    sizea
    dtype
end

function FD(sizea, dtype; set=true)
    d = Cptr[0]
    @cuda(cudnn,cudnnCreateFilterDescriptor,(Ptr{Cptr},),d)
    n = length(sizea)
    sz = [Cint(sizea[n-i+1]) for i=1:n]
    if cudnnVersion >= 5000
        @cuda(cudnn,cudnnSetFilterNdDescriptor,
              (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
              d[1], dtype, 0,     n,   sz)
    elseif cudnnVersion >= 4000
        @cuda(cudnn,cudnnSetFilterNdDescriptor_v4,
              (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
              d[1], dtype, 0,     n,   sz)
    else
        @cuda(cudnn,cudnnSetFilterNdDescriptor,
              (Cptr,UInt32,Cint,Ptr{Cint}),
              d[1], dtype,    n,   sz)
        fd = FD(d[1], sizea, dtype)
    end
    finalizer(fd, x->@cuda(cudnn,cudnnDestroyFilterDescriptor,(Cptr,),x.ptr))
    return fd
end

#only initialize
function FD()
    d = Cptr[0]
    @cuda(cudnn,cudnnCreateFilterDescriptor,(Ptr{Cptr},),d)
    fd = FD(d[1], nothing, nothing)
    finalizer(fd, x->@cuda(cudnn,cudnnDestroyFilterDescriptor,(Cptr,),x.ptr))
    return fd
end

function FD(a::KnetArray)
    eltype = eltype(a)
    dtype = DT(eltype)
    s = size(a)
    if ndims(a) == 1 #rnn accepts 3d tensors
        s = (1, 1, length(a))
    end
    return FD(s, dtype)
end



import Base.unsafe_convert
unsafe_convert(::Type{Cptr}, td::TD)=td.ptr
unsafe_convert(::Type{Cptr}, td::RD)=td.ptr
unsafe_convert(::Type{Cptr}, td::DD)=td.ptr


# The workspace abstraction
let
    # TODO: make this shared with cnns?
    wsdict = Dict{Integer, Any}()

    # only weight backward will be enoguh due to caching
    global getws, cleanws!, wssizes
    
    function getws(rd, xtds;o...)
        wss = workspace_size(cache.rd, xtds; o...)
        return haskey(wsdict, wss) ? wsdict[wss] : (wsdict[wss]=KnetArray{Int8}(wss))
    end

    function cleanws!()
        for k in keys(wsdict)
            delete!(wsdict, k)
        end
    end

    wssizes() = [k for k in keys(wsdict)]
end
