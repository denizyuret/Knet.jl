module KUnet
using InplaceOps
using CUDArt
using Base.LinAlg.BLAS
using CUBLAS
export Layer, Net, setparam!

type Layer w; b; fx; fy; dw; db; pw; pb; y; x; dx; dropout; xdrop; Layer()=new() end
typealias Net Array{Layer,1}
type UpdateParam 
    learningRate
    l1reg
    l2reg
    maxnorm
    adagrad;  ada
    momentum; mom
    nesterov; nes
    UpdateParam()=new(0.01)
end

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

include("layer.jl")
include("net.jl")
include("cuda.jl")
include("update.jl")
include("func.jl")
include("h5io.jl")

end # module
