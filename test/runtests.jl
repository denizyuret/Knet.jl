using KUnet
using Base.Test
using Base.Test: Success, Failure, Error
import Base.Test: default_handler
if KUnet.GPU
    eval(Expr(:using,:CUDArt))
    eval(Expr(:using,:CUDArt,:ContiguousArray))
else
    typealias ContiguousArray{T} Array{T}
end

# Uncomment this if you want lots of messages:
# default_handler(r::Success) = info("$(r.expr)")
# default_handler(r::Failure) = warn("FAIL: $(r.expr)")
# default_handler(r::Error)   = warn("$(r.err): $(r.expr)")

function Base.isapprox(x::ContiguousArray,y::ContiguousArray;
                       maxeps::Real = max(eps(eltype(x)), eps(eltype(y))),
                       rtol::Real=cbrt(maxeps), atol::Real=sqrt(maxeps))
    size(x) == size(y) || (warn("isapprox: $(size(x))!=$(size(y))"); return false)
    x = to_host(x)
    y = to_host(y)
    d = abs(x-y)
    s = abs(x)+abs(y)
    all(d .< (atol + rtol * s))
end

function gradcheck(net, x1, z1, w, dw; iter=10, 
                   epsilon=cbrt(eps(eltype(w))), 
                   delta=(eltype(w)==Float64) ? 1e-4 : 1e-3)
    maxdiff = 0.0
    for i=1:iter
        r = (iter==length(w) ? i : rand(1:length(w)))
        wr0 = w[r]
        wr1 = wr0 - delta
        wr2 = wr0 + delta
        # Do not cross 0 for softloss
        (wr0>0) && (wr1<0) && (wr1=0)
        (wr0<0) && (wr2>0) && (wr2=0)
        w[r] = wr1; loss1 = getloss(net, x1, z1)
        w[r] = wr2; loss2 = getloss(net, x1, z1)
        w[r] = wr0
        dwr = (loss2 - loss1) / (wr2 - wr1)
        # @show (dw[r], dwr)
        absdiff = abs(dwr - dw[r])/(abs(dwr) + abs(dw[r]))
        absdiff > maxdiff && (maxdiff = absdiff)
    end
    # @show (maxdiff, epsilon, delta)
    return maxdiff < epsilon
end

function forwlossback(net, x, z)
    n = length(net)
    xx = Any[]
    for i=1:n
        x = forw(net[i], copy(x); seed=1)
        push!(xx, x)
    end
    ll = (isa(net[n], LossLayer) ? loss(net[n],z) : 0)
    zz = Any[]
    for i=n:-1:1
        z = back(net[i], copy(z))
        unshift!(zz, z)
    end
    return (ll, xx, zz)
end

function getloss(net, x, z)
    x = (isa(net[1],LogpLoss) ? forw(Logp(),copy(x)) :
         isa(net[1],SoftLoss) ? (copy(x)./sum(x,1)) : copy(x))
    # @show x
    n = length(net)
    for i=1:n; x = forw(net[i], x; seed=1); end
    loss(net[n], z)
end

function getparam(l1::Layer)
    w1 = nothing
    for n in names(l1); isdefined(l1,n) && isa(l1.(n), Param) && (w1=l1.(n); break); end
    return w1
end

function gputest(cnet::Net, x, z)
    rval = true
    # Compare loss, y, dx, dw after forw and back:
    (cl, cx, cz) = forwlossback(cnet, x, z)
    KUnet.atype(CudaArray)
    gnet = copy(cnet)
    (gl, gx, gz) = forwlossback(gnet, CudaArray(x), CudaArray(z))
    isapprox(gl, cl) || (warn("loss mismatch in $(map(typeof,cnet))"); rval=false)
    for i=1:length(cnet)
        isapprox(cx[i], gx[i]) || (warn("y mismatch in $(typeof(cnet[i]))"); rval=false)
        isapprox(cz[i], gz[i]) || (warn("dx mismatch in $(typeof(cnet[i]))"); rval=false)
        cw = getparam(cnet[i]); gw = getparam(gnet[i])
        cw == nothing || isapprox(cw.diff, gw.diff) || (warn("dw mismatch in $(typeof(cnet[i]))"); rval=false)
    end
    KUnet.atype(Array)

    # Compare w, dw after update:
    setparam!(cnet; lr=0.1, l1reg=0.1, l2reg=0.1, adagrad=0.1, momentum=0.1, nesterov=0.1)
    setparam!(gnet; lr=0.1, l1reg=0.1, l2reg=0.1, adagrad=0.1, momentum=0.1, nesterov=0.1)
    update(cnet); update(gnet)
    for i=1:length(cnet)
        cw = getparam(cnet[i]); gw = getparam(gnet[i])
        cw == nothing && continue
        isapprox(cw.diff, gw.diff) || (warn("dw mismatch after update in $(typeof(cnet[i]))"); rval=false)
        isapprox(cw.data, gw.data) || (warn("w mismatch after update in $(typeof(cnet[i]))"); rval=false)
    end

    return rval
end

function gradtest(net, x, z)
    rval = true
    (ll, yy, dx) = forwlossback(net, x, z)
    gradcheck(net, x, z, x, dx[1]) || (warn("gradtest failed for dx in $(typeof(net[1]))"); rval=false)
    w1 = getparam(net[1])
    w1 == nothing || gradcheck(net, x, z, w1.data, w1.diff) || (warn("gradtest failed for dw in $(typeof(net[1]))"); rval=false)
    return rval
end

function iseq(a,b)
    typeof(a)==typeof(b) || return false
    isa(a,Tuple) && return all(map(iseq, a, b))
    isempty(names(a)) && return isequal(a,b)
    for n in names(a)
        in(n, (:x,:y,:dx,:dy,:xdrop)) && continue
        in(n, names(b)) || (warn("$n missing");return false)
        isdefined(a,n) || continue
        isdefined(b,n) || (warn("$n undefined");return false)
        iseq(a.(n), b.(n)) || (warn("$n unequal"); return false)
    end
    for n in names(b)
        in(n, (:x,:y,:dx,:dy,:xdrop)) && continue
        in(n, names(a)) || (warn("$n missing");return false)
        isdefined(b,n) || continue
        isdefined(a,n) || (warn("$n undefined");return false)
    end
    return true
end

function filetest(net1)
    KUnet.savenet("/tmp/kunet.test", net1)
    net2 = KUnet.loadnet("/tmp/kunet.test")
    @assert all(map(iseq, net1, net2))
end

function getz(net, x)
    (net == nothing || x == nothing) && return nothing
    z = rand!(forw(net, copy(x)))
    L = typeof(net[end])
    return (in(L, (Logp,)) ? forw(Logp(), z) :
            in(L, (Soft, SoftLoss, LogpLoss, XentLoss)) ? forw(Soft(), z) : z)
end

function getnet{T<:Layer}(F,S,L::Type{T})
    nd = length(S)
    nf = (nd==1 ? S[1] : div(prod(S),S[nd]))
    (nf>20) && in(L,(Logp,Soft,LogpLoss,SoftLoss,XentLoss)) && return nothing
    (nd!=4) && in(L,(Conv,Pool)) && return nothing
    C = (nd==1 ? S[1] : S[nd-1])
    l = ((L == Bias) ? Bias(Param(rand(F, C))) :
         (L == Conv) ? Conv(rand(1:S[1]),rand(1:S[2]),C,rand(1:20)) :
         (L == Drop) ? Drop(rand()) :
         (L == Mmul) ? Mmul(rand(1:20), nf) :
         (L == Pool) ? Pool(rand(1:minimum(S))) :
         (L == Logp) ? Logp() :
         (L == Soft) ? Soft() : L())
    net = Layer[]; push!(net, l)
    return (isa(l, Logp) ? push!(net, LogpLoss()) :
            isa(l, Soft) ? push!(net, SoftLoss()) :
            !isa(l, LossLayer) ? push!(net, QuadLoss()) : net)
end

function getx(F,S,L)
    return ((L == LogpLoss) ? forw(Logp(), rand(F,S)) :
            (L == SoftLoss) ? forw(Soft(), rand(F,S)) :
            rand(F, S))
end

function gettest(F,S,L)
    net = getnet(F,S,L)
    x = getx(F,S,L)
    z = getz(net, x)
    return (net, x, z)
end

net0 = x0 = z0 = nothing

function main(layers)
    global net0, x0, z0
    KUnet.atype(Array)
    for F in (Float32,Float64)
        KUnet.ftype(F)
        for D in 1:5
            S = tuple(rand(1:20,D)...)
            for L in layers
                (net, x, z) = gettest(F,S,L)
                net==nothing && continue  # combination not supported
                net0, x0, z0 = net, x, z
                @show (F, S, L)
                KUnet.GPU && gputest(net, x, z)
                gradtest(net, x, z)
                filetest(net)
            end
        end
    end
end


# Test each layer for: 1D-5D, gpu/cpu, float32/float64
# layers = (Bias, Conv, Drop, Logp, LogpLoss, Mmul, PercLoss, Pool, QuadLoss, Relu, Sigm, Soft, SoftLoss, Tanh, XentLoss)
layers=(PercLoss,)
main(layers)

