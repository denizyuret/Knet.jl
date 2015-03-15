# Layer: both f (activation function) and b (bias vector) are
# optional, w (weight matrix) is the only required field

type Layer w; b; f; fx; dw; db; pw; pb; y; x; dx; dropout; xdrop; Layer()=new() end
atype(w) = (usegpu ? CudaArray(w) : w)
Layer(w; o...) = (l=Layer(); l.w=atype(w); setparam!(l; o...); l)
Layer(w, b; o...) = (l=Layer(); l.w=atype(w); l.b=atype(b); setparam!(l; o...); l)
Layer(f::Function, w; o...) = (l=Layer(w; o...); l.f=f; l)
Layer(f::Function, w, b; o...) = (l=Layer(w,b; o...); l.f=f; l)
Layer(c::Integer, r::Integer; bias=true, o...) = (w=float32(randn(r,c)*0.01);bias ? Layer(w,zeros(Float32, r, 1);o...) : Layer(w; o...))
Layer(f::Function, c::Integer, r::Integer; o...) = (l=Layer(c,r;o...); l.f=f; l)
istransient(l,n)=(isa(l,Layer) && in(n,(:y,:x,:dx,:xdrop)))  # no need to copy or save these

# Net: Convenience type and constructor for an array of layers

typealias Net Array{Layer,1}
# Julia v0.4 allows Net as a constructor name, but v0.3 does not:
# Net(f::Function, d::Integer...; o...) = (n=Layer[]; for i=2:length(d); push!(n, (i<length(d)) ? Layer(f,d[i-1],d[i];o...) : Layer(d[i-1],d[i];o...)); end; n)
newnet(f::Function, d::Integer...; o...) = (n=Layer[]; for i=2:length(d); push!(n, (i<length(d)) ? Layer(f,d[i-1],d[i];o...) : Layer(d[i-1],d[i];o...)); end; n)


# UpdateParam: Parameters can be set at the level of weights, layers, nets, or
# training sessions.  Here is the lowest level each parameter can be
# set:
# weight-specific: learningRate; l1reg; l2reg; maxnorm; adagrad; momentum; nesterov; 
# layer-specific: dropout, apply_fx, return_dx
# net-specific: ? 
# train-specific: batch, iters, loss

type UpdateParam learningRate; l1reg; l2reg; maxnorm; adagrad; ada; momentum; mom; nesterov; nes; UpdateParam(lr)=new(lr); end

function UpdateParam(; learningRate=0.01f0, args...)
    o=UpdateParam(learningRate)
    setparam!(o; args...)
    return o
end

setparam!(x; args...)=(for (k,v)=args; setparam!(x,k,v); end)
setparam!(p::UpdateParam,k,v)=(p.(k)=convert(Float32, v))
setparam!(net::Net,k,v)=for l=net; setparam!(l,k,v) end

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


# Just a convenience type for training etc.
type XY x; y; XY()=new(); end

