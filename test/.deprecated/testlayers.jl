using Compat
using Knet
using Base.Test
using Base.Test: Success, Failure, Error
import Base.Test: default_handler
include("isapprox.jl")
if Knet.GPU
    eval(Expr(:using,:CUDArt))
    eval(Expr(:using,:CUDArt,:ContiguousArray))
else
    typealias ContiguousArray{T} Array{T}
end

gnet0 = net0 = x0 = z0 = nothing

# Uncomment this if you want lots of messages:
# default_handler(r::Success) = info("$(r.expr)")
# default_handler(r::Failure) = warn("FAIL: $(r.expr)")
# default_handler(r::Error)   = warn("$(r.err): $(r.expr)")

function gradcheck(net, x1, z1, w, dw; iter=10, 
                   epsilon=eps(eltype(w))^(1/4), 
                   delta=(eltype(w)==Float64) ? 1e-4 : 1e-3)
    for i=1:iter
        r = (iter==length(w) ? i : rand(1:length(w)))
        wr0 = w[r]
        wr1 = wr0 - delta
        wr2 = wr0 + delta
        # Do not cross 0 for softloss
        (wr0>=0) && (wr1<0) && (wr1=0)
        (wr0<0) && (wr2>0) && (wr2=0)
        w[r] = wr1; loss1 = getloss(net, x1, z1)
        w[r] = wr2; loss2 = getloss(net, x1, z1)
        w[r] = wr0
        dwr = (loss2 - loss1) / (wr2 - wr1)
        # @show (dw[r], dwr)
        if !isapprox(dw[r], dwr; rtol=0.1)
            @show (:gradcheck, dw[r], dwr)
            return false
        end
    end
    return true
end

function forwlossback(net, x, z)
    n = length(net)
    xx = Any[]
    for i=1:n
        x = forw(net[i], copy(x); seed=1)
        @assert isa(x, KUdense)
        push!(xx, x)
    end
    ll = (isa(net[n], Loss) ? loss(net[n],z) : 0)
    zz = Any[]
    for i=n:-1:1
        z = back(net[i], copy(z))
        @assert isa(z, KUdense)
        unshift!(zz, z)
    end
    return (ll, xx, zz)
end

function getloss(net, x, z)
    x = copy(x)
    isa(net[1],LogpLoss) && (x=forw(Logp(),x))
    isa(net[1],SoftLoss) && (x=KUdense(x.arr ./ sum(x.arr,1)))
    # @show x
    n = length(net)
    for i=1:n; x = forw(net[i], x; seed=1); @assert isa(x, KUdense); end
    loss(net[n], z)
end

function getparam(l1::Op)
    w1 = nothing
    for n in fieldnames(l1); isdefined(l1,n) && isa(l1.(n), KUparam) && (w1=l1.(n); break); end
    return w1
end

function gputest(cnet::MLP, x, z)
    global gnet0
    rval = true
    # Compare loss, y, dx, dw after forw and back:
    # info("gputest 1")
    # display(shownet(cnet));println("")
    (cl, cx, cz) = forwlossback(cnet, x, z)
    # info("gputest 2")
    # display(shownet(cnet));println("")
    gnet0 = gnet = Op[gpucopy(cnet)...]
    # info("gputest 3")
    # display(shownet(gnet));println("")
    # @show gnet
    # hnet = Op[cpucopy(gnet)...]
    # @show hnet
    # @assert isequal(cnet[1].w.arr, hnet[1].w.arr)
    # @show cnet
    xx,zz = gpucopy(x),gpucopy(z)
    # @show (xx,zz)
    (gl, gx, gz) = forwlossback(gnet, xx, zz)
    # info("gputest 4")
    # display(shownet(gnet));println("")
    isapprox(gl, cl; rtol=cbrt(eps(Float32)), atol=sqrt(eps(Float32))) || (warn("loss mismatch in $(map(typeof,cnet)): $gl != $cl"); rval=false)
    # info("gputest 5")
    for i=1:length(cnet)
        isapprox(cx[i], gx[i]) || (warn("y mismatch in $(typeof(cnet[i]))"); rval=false)
        isapprox(cz[i], gz[i]) || (warn("dx mismatch in $(typeof(cnet[i]))"); rval=false)
        cw = getparam(cnet[i]); gw = getparam(gnet[i])
        cw == nothing || isapprox(cw.diff, gw.diff) || (warn("dw mismatch in $(typeof(cnet[i]))"); rval=false)
    end

    # Compare w, dw after update:
    # info("gputest 6")
    setparam!(cnet; lr=0.1, l1reg=0.1, l2reg=0.1, adagrad=true, momentum=0.1, nesterov=0.1)
    # info("gputest 7")
    setparam!(gnet; lr=0.1, l1reg=0.1, l2reg=0.1, adagrad=true, momentum=0.1, nesterov=0.1)
    # info("gputest 8")
    update!(cnet)
    # info("gputest 9")
    update!(gnet)
    # info("gputest 10")
    for i=1:length(cnet)
        cw = getparam(cnet[i]); gw = getparam(gnet[i])
        cw == nothing && continue
        isapprox(cw.diff, gw.diff) || (warn("dw mismatch after update in $(typeof(cnet[i]))"); rval=false)
        isapprox(cw.arr, gw.arr) || (warn("w mismatch after update in $(typeof(cnet[i]))"); rval=false)
    end
    # info("gputest 11")

    return rval
end

function gradtest(net, x, z)
    rval = true
    (ll, yy, dx) = forwlossback(net, x, z)
    gradcheck(net, x, z, x, dx[1]) || (warn("gradtest failed for dx in $(typeof(net[1]))"); rval=false)
    w1 = getparam(net[1])
    w1 == nothing || gradcheck(net, x, z, w1.arr, w1.diff) || (warn("gradtest failed for dw in $(typeof(net[1]))"); rval=false)
    return rval
end

function iseq03(a,b)
    typeof(a)==typeof(b) || return false
    isa(a,Tuple) && return all(map(iseq03, a, b))
    isempty(fieldnames(a)) && return isequal(a,b)
    for n in fieldnames(a)
        in(n, (:x,:y,:dx,:dy,:xdrop)) && continue
        in(n, fieldnames(b)) || (warn("$n missing");return false)
        isdefined(a,n) || continue
        isdefined(b,n) || (warn("$n undefined");return false)
        iseq03(a.(n), b.(n)) || (warn("$n unequal"); return false)
    end
    for n in fieldnames(b)
        in(n, (:x,:y,:dx,:dy,:xdrop)) && continue
        in(n, fieldnames(a)) || (warn("$n missing");return false)
        isdefined(b,n) || continue
        isdefined(a,n) || (warn("$n undefined");return false)
    end
    return true
end

function filetest(net1)
    isa(net1[1], Pool) && (warn("Pooling layers cannot be saved to file yet"); return true)
    Knet.savenet("/tmp/knet.test", net1)
    net2 = Knet.loadnet("/tmp/knet.test")
    return all(map(iseq03, net1, net2))
end

function getnet{T<:Op}(F,S,L::Type{T})
    nd = length(S)
    nf = (nd==1 ? S[1] : div(prod(S),S[nd]))
    (nf>20) && in(L,(Logp,Soft,LogpLoss,SoftLoss,XentLoss)) && return nothing
    (nd!=4) && in(L,(Conv,Pool)) && return nothing
    C = (nd==1 ? S[1] : S[nd-1])
    l = ((L == Conv) ? Conv(rand(1:20), rand(1:min(S[1],S[2]))) :
         (L == Drop) ? Drop(rand()) :
         (L == Mmul) ? Mmul(rand(1:20)) :
         (L == Pool) ? Pool(rand(1:minimum(S))) : L())
    net = Op[]; push!(net, l)
    return (isa(l, Logp) ? push!(net, LogpLoss()) :
            isa(l, Soft) ? push!(net, SoftLoss()) :
            !isa(l, Loss) ? push!(net, QuadLoss()) : net)
end

function getx(F,S,L)
    x = KUdense(rand(F,S))
    return ((L == LogpLoss) ? forw(Logp(), x) :
            (L == SoftLoss) ? forw(Soft(), x) : x)
end

function getz(net, x)
    (net == nothing || x == nothing) && return nothing
    z = copy(rand!(forw(net, copy(x))))
    @assert isa(z, KUdense)
    L = typeof(net[end])
    return (in(L, (Logp,)) ? forw(Logp(), z) :
            in(L, (Soft, SoftLoss, LogpLoss, XentLoss)) ? forw(Soft(), z) : z)
end

function gettest(F,S,L)
    net = getnet(F,S,L)
    x = getx(F,S,L)
    z = getz(net, x)
    return (net, x, z)
end

function shownet(n::MLP)
    map(showlayer, n)
end

function showlayer(l::Op)
    ans = Any[]
    push!(ans,typeof(l))
    for n in fieldnames(l)
        isdefined(l,n) || continue
        f = l.(n)
        push!(ans,n)
        # push!(ans,typeof(f))
        if isa(f, KUdense)
            # @assert pointer(f) === pointer(f.arr) === pointer(f.ptr)
            push!(ans, pointer(f.arr))
        elseif isa(f, KUparam)
            push!(ans, pointer(f.arr))
        end
    end
    ans
end

function main(layers)
    global net0, x0, z0
    Knet.gpu(false)
    for F in (Float32,Float64)
        for D in 1:5
            S = tuple(rand(1:20,D)...)
            for L in layers
                (net, x, z) = gettest(F,S,L)
                net==nothing && continue  # combination not supported
                net0, x0, z0 = net, x, z
                @show (F, S, L)
                Knet.GPU && (@test gputest(net, x, z))
                gradtest(net, x, z)
                @test filetest(net)
            end
        end
    end
end

# Test each layer for: 1D-5D, gpu/cpu, float32/float64
layers = (Bias, Conv, Drop, Logp, LogpLoss, Mmul, PercLoss, Pool, QuadLoss, Relu, Sigm, Soft, SoftLoss, Tanh, XentLoss)
# These don't have cpu versions: Conv, Pool
# layers = (Bias, Drop, Logp, LogpLoss, Mmul, PercLoss, QuadLoss, Relu, Sigm, Soft, SoftLoss, Tanh, XentLoss)
# layers = (Conv,Pool)
main(layers)

