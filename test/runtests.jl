using Base.Test
using CUDArt
using KUnet
using CUDArt: ContiguousArray
using Base.Test: Success, Failure, Error
import Base.Test: default_handler
using KUnet: size2

# Uncomment this if you want lots of messages:
default_handler(r::Success) = info("$(r.expr)")
default_handler(r::Failure) = warn("FAIL: $(r.expr)")
default_handler(r::Error)   = warn("$(r.err): $(r.expr)")

function Base.isapprox(x::ContiguousArray,y::ContiguousArray;
                       maxeps::Real = max(eps(eltype(x)), eps(eltype(y))),
                       rtol::Real=cbrt(maxeps), atol::Real=sqrt(maxeps))
    size(x) == size(y) || (warn("isapprox: $(size(x))!=$(size(y))"); return false)
    x,y = to_host(x), to_host(y)
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

CudaLayer(l)=(isa(l, Mmul) ? Mmul(CudaArray(l.w.data)) :
              isa(l, Conv) ? Conv(CudaArray(l.w.data)) :
              isa(l, Bias) ? Bias(CudaArray(l.b.data)) : 
              isa(l, Drop) ? Drop(l.dropout) :
              isa(l, Pool) ? Pool(l.pd) :
              typeof(l)())

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
    for n in names(l1); isa(l1.(n), Param) && (w1=l1.(n); break); end
    return w1
end

function gputest(cnet::Net, x, z)
    rval = true
    KUnet.GPU || (warn("GPU not available"); return false)
    (cl, cx, cz) = forwlossback(cnet, x, z)
    KUnet.atype(CudaArray)
    gnet = map(CudaLayer, cnet)
    (gl, gx, gz) = forwlossback(gnet, CudaArray(x), CudaArray(z))
    isapprox(gl, cl) || (warn("loss mismatch in $(map(typeof,cnet))"); rval=false)
    for i=1:length(cnet)
        isapprox(cx[i], gx[i]) || (warn("y mismatch in $(typeof(cnet[i]))"); rval=false)
        isapprox(cz[i], gz[i]) || (warn("dx mismatch in $(typeof(cnet[i]))"); rval=false)
        cw = getparam(cnet[i]); gw = getparam(gnet[i])
        cw == nothing || isapprox(cw.diff, gw.diff) || (warn("dw mismatch in $(typeof(cnet[i]))"); rval=false)
    end
    KUnet.atype(Array)
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

function getz(net, x)
    z = rand!(forw(net, copy(x)))
    L = typeof(net[end])
    return (in(L, (Logp,)) ? forw(Logp(), z) :
            in(L, (Soft, SoftLoss, LogpLoss)) ? forw(Soft(), z) : z)
end

function getnet{T<:Layer}(F,S,L::Type{T})
    nd = length(S)
    nf = (nd==1 ? S[1] : div(prod(S),S[nd]))
    C = (nd==1 ? S[1] : S[nd-1])
    l = ((L == Bias) ? Bias(Param(rand(F, C))) :
         (L == Conv) ? Conv(rand(1:S[1]),rand(1:S[2]),C,rand(1:20)) :
         (L == Drop) ? Drop(rand()) :
         (L == Mmul) ? Mmul(rand(1:20), nf) :
         (L == Pool) ? Pool(rand(1:minimum(S))) :
         (L == Logp) ? Logp() :
         (L == Soft) ? Soft() : L())
    net = Layer[l]
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

net3 = x3 = z3 = nothing

function main(layers)
    KUnet.atype(Array)
    # for F in (Float32,Float64)
    for F in (Float64,)
        KUnet.ftype(F)
        #    for D in 1:5
        for D in 1:1
            S = tuple(rand(1:20,D)...)
            for L in layers
                @show (F, S, L)
                (net, x, z) = gettest(F,S,L)
                net==nothing && (warn("Not supported"); continue)
                gputest(net, x, z)
                gradtest(net, x, z)
                global net3 = net
                global x3 = x
                global z3 = z
            end
        end
    end
end


# Test each layer for: 1D-5D, gpu/cpu, float32/float64
# layers = (Bias, Conv, Drop, Logp, LogpLoss, Mmul, Pool, QuadLoss, Relu, Sigm, Soft, SoftLoss, Tanh, XentLoss)
# failed64gpu = (Conv, Pool, XentLoss)
# passed64gpu = (Bias, Drop, Logp, LogpLoss, QuadLoss, Relu, Sigm, Soft, SoftLoss, Tanh)
layers = (Mmul,)
main(layers)

#             qloss1 = QuadLoss()
#             y1 = forw(l1, copy(x1); xdrop=xdrop)
#             forw(qloss1, y1)  # network output
#             z1 = rand(ft, size(y1))  # desired answers
#             loss1 = loss(qloss1, z1)  # loss value
#             dy1 = back(qloss1, copy(z1))
#             dx1 = back(l1, copy(dy1))

#             if KUnet.GPU
#                 KUnet.atype(CudaArray)
#                 qloss2 = QuadLoss()
#                 l2 = CudaLayer(l1)
#                 x2 = CudaArray(x1)
#                 y2 = forw(l2, copy(x2); xdrop=xdrop)
#                 @test isapprox(y1, y2)
#                 forw(qloss2, y2)
#                 z2 = CudaArray(z1)
#                 loss2 = loss(qloss2, z2)
#                 # @show (loss1, loss2)
#                 @test isapprox(loss1, loss2)
#                 dy2 = back(qloss2, copy(z2))
#                 dx2 = back(l2, copy(dy2))
#                 @test isapprox(dx1, dx2)
#                 KUnet.atype(Array)
#             end

#             # Gradient check:
#             lossfn1=()->loss(qloss1,z1)
#             @test gradcheck(y1, dy1, lossfn1)
#             lossfn2=()->(forw(qloss1,forw(l1,copy(x1);xdrop=xdrop));loss(qloss1,z1))
#             @test gradcheck(x1, dx1, lossfn2)
#             w1 = nothing
#             for n in names(l1); isa(l1.(n), Param) && (w1=l1.(n)); end
#             if w1 != nothing
#                 @test gradcheck(w1.data, w1.diff, lossfn2)
#             end  # if w1 != nothing
#         end # for l1

#         for l1 in (
#                    # LogpLoss(),
#                    # QuadLoss(),
#                    # SoftLoss(),
#                    # XentLoss(),
#                    )
#             if isa(l1, QuadLoss)
#                 y1 = x1
#                 z1 = rand(ft, size(y1))
#                 @show (ft, nd, size(y1), typeof(l1))
#             else
#                 nd > 2 && continue  # cross entropy losses are accurate for a few classes
#                 y1 = (nd==1 ? rand(ft, 10) : rand(ft, 10, size(x1,nd)))
#                 z1 = forw(Soft(), 5*rand(ft, size(y1)))
#                 @show (ft, nd, size(y1), typeof(l1))
#                 nxones = ones(ft, (nd==1 ? 1 : size(x1, nd)))
#                 @test isapprox(nxones, vec(sum(z1,1)))
#                 if isa(l1, LogpLoss) 
#                     forw(Logp(), y1)
#                     @test isapprox(nxones, vec(sum(exp(y1),1)))
#                 end
#                 if isa(l1, SoftLoss)
#                     forw(Soft(), y1)
#                     @test isapprox(nxones, vec(sum(y1,1)))
#                 end
#             end

#             forw(l1, y1)
#             dy1 = back(l1, copy(z1))
            
#             if KUnet.GPU
#                 KUnet.atype(CudaArray)
#                 l2 = CudaLayer(l1)
#                 y2 = CudaArray(y1)
#                 z2 = CudaArray(z1)
#                 forw(l2,y2)
#                 dy2 = back(l2, copy(z2))
#                 @test isapprox(dy1,dy2)
#                 KUnet.atype(Array)
#             end

#             lossfn3=()->(isa(l1,LogpLoss) ? forw(l1,forw(Logp(),copy(y1))) :
#                          isa(l1,SoftLoss) ? forw(l1,copy(y1)./sum(y1,1)) :
#                          forw(l1,y1); loss(l1,z1))
#             @test gradcheck(y1, dy1, lossfn3)

#         end # for l1 in lossfns

#         for l1 in (             # no CPU impl or gradcheck
#                    # Conv(5,5,xchan,10), # only gpu, only 4D
#                    # Pool(2, max(1,nd-2)), # only gpu, only 4D
#                    )
#             nd != 4 && continue
#             @show (ft, nd, size(x1), typeof(l1))
#             KUnet.atype(CudaArray)
#             qloss2 = QuadLoss()
#             l2 = CudaLayer(l1)
#             x2 = CudaArray(x1)
#             y2 = forw(l2, copy(x2); xdrop=xdrop); @test y2!=nothing
#             # @test isapprox(y1, y2)
#             y3 = forw(qloss2, y2); @test y3!=nothing
#             # z2 = CudaArray(z1)
#             z2 = CudaArray(rand(ft, size(y2)))
#             @show loss2 = loss(qloss2, z2)
#             # @show (loss1, loss2)
#             # @test isapprox(loss1, loss2)
#             dy2 = back(qloss2, copy(z2)); @test dy2!=nothing
#             dx2 = back(l2, copy(dy2)); @test dx2!=nothing
#             # @test isapprox(dx1, dx2)
#             KUnet.atype(Array)
#         end # for l1 in conv,pool

#     end # for nd
# end # for ft


