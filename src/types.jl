# export Layer, Net, UpdateParam, setparam!

type Layer f; w; b; fx; dw; db; pw; pb; y; x; dx; dropout; xdrop;  # can we use dx for xdrop?
    function Layer(; args...)
        o=new()
        for (k,v)=args
            in(k, names(o)) ? (o.(k) = v) : warn("Layer has no field $k")
        end
        return o
    end
end

type UpdateParam learningRate; l1reg; l2reg; maxnorm; adagrad; ada; momentum; mom; nesterov; nes; 
    function UpdateParam(; learningRate=0.01f0, args...)
        o=new(learningRate)
        for (k,v)=args
            in(k, names(o)) ? (o.(k) = v) : warn("UpdateParam has no field $k")
        end
        return o
    end
end

typealias Net Array{Layer,1}

function Net(fn::Function, dims::Integer...)
    net = Layer[]
    for i=2:length(dims)
        nrows,ncols = dims[i],dims[i-1]
        l = (i < length(dims)) ? Layer(nrows, ncols; f=fn) : Layer(nrows, ncols)
        push!(net, l)
    end
    return net
end

function Layer(nrows::Integer, ncols::Integer; args...)
    l = Layer(; args...)
    arr = usegpu ? CudaArray : Array
    l.w = arr(Float32, nrows, ncols)
    l.b = arr(Float32, nrows, 1)
    rand!(l.w); @in1! l.w .- 0.5f0; @in1! l.w .* 0.05f0;
    fill!(l.b, 0f0)
    return l
end

# Parameters can be set at the level of weights, layers, nets, or
# training sessions.  Here is the lowest level each parameter can be
# set:
# weight-specific: learningRate; l1reg; l2reg; maxnorm; adagrad; momentum; nesterov; 
# layer-specific: dropout, apply_fx, return_dx
# net-specific: ? 
# train-specific: batch, iters, loss

function setparam!(l::Layer,k,v)
    if (k == :dropout)
        l.dropout = v
        (v > zero(v)) && (l.fx = drop)
        return
    end
    if isdefined(l, :w)
        isdefined(l, :pw) || (l.pw = UpdateParam())
        setparam!(l.pw, k, v)
    end
    if isdefined(l, :b)
        isdefined(l, :pb) || (l.pb = UpdateParam())
        if in(k, [:l1reg, :l2reg, :maxnorm]) && (v != zero(v))
            warn("Skipping $k regularization for bias.")
        else
            setparam!(l.pb, k, v)
        end
    end
end

setparam!(p::UpdateParam,k,v)=(p.(k)=v)

setparam!(net::Net,k,v)=for l=net setparam!(l,k,v) end

