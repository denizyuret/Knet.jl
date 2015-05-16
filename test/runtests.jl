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
    # @show maximum(d)
    # @show maximum(d./s)
    (maximum(d) <= atol) && (maximum(d./s) <= rtol)
end

function gradcheck(net, x1, z1, x, dx; iter=10, 
                   epsilon=cbrt(eps(eltype(x))), 
                   delta=(eltype(x)==Float64) ? 1e-4 : 1e-3)
    maxdiff = 0.0
    for i=1:iter
        r = (iter==length(x) ? i : rand(1:length(x)))
        xr0 = x[r]
        # Do not cross 0 for softloss
        xr1 = xr0 - delta; xr0>0 && xr1<0 && (xr1=0)
        xr2 = xr0 + delta; xr0<0 && xr2>0 && (xr2=0)
        x[r] = xr1; loss1 = getloss(net, x1, z1)
        x[r] = xr2; loss2 = getloss(net, x1, z1)
        x[r] = xr0
        dxr = (loss2 - loss1) / (xr2 - xr1)
        @show (dx[r], dxr)
        absdiff = abs(dxr - dx[r])/(abs(dxr) + abs(dx[r]))
        absdiff > maxdiff && (maxdiff = absdiff)
    end
    @show (maxdiff, epsilon, delta)
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
    n = length(net)
    for i=1:n; x = forw(net[i], copy(x); seed=1); end
    loss(net[n], z)
end

function getparam(l1::Layer)
    w1 = nothing
    for n in names(l1); isa(l1.(n), Param) && (w1=l1.(n); break); end
    return w1
end

function getlayer(lt, x)
    nd = ndims(x)
    (x1,x2) = KUnet.size2(x)
    nc = (nd==1 ? x1 : size(x, nd-1))
    lt == Bias ? Bias(Param(rand(eltype(x), nc))) :
    lt == Conv ? Conv(rand(1:size(x,1)),rand(1:size(x,2)),nc,rand(1:20)) :
    lt == Drop ? Drop(rand()) :
    lt == Mmul ? Mmul(rand(1:20),x1) :
    lt == Pool ? Pool(rand(1:minimum(size(x)))) :
    lt()
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


# Test each layer for: 1D-5D, gpu/cpu, float32/float64
# layers = (Bias, Conv, Drop, Logp, LogpLoss, Mmul, Pool, QuadLoss, Relu, Sigm, Soft, SoftLoss, Tanh, XentLoss)
# failed64gpu = (Conv, Logp, Mmul, Pool, XentLoss)
# passed64gpu = (Bias, Drop, LogpLoss, QuadLoss, Relu, Sigm, Soft, SoftLoss, Tanh)
layers = (Sigm,)
net = x1 = z1 = nothing

KUnet.atype(Array)
# for F in (Float32,Float64)
for F in (Float64,)
    KUnet.ftype(F)
#    for D in 1:5
    for D in 1:1
        x1 = rand(F, tuple(rand(1:20,D)...))
        S = size(x1)
        for L in layers
            @show (F, D, S, L)
            l1 = getlayer(L, x1)
            l1 == nothing && (warn("Not supported"); continue)
            net = Layer[l1]
            (L <: LossLayer) || push!(net, QuadLoss())
            x1 = rand!(x1)
            z1 = rand!(forw(net, copy(x1)))
            gputest(net, x1, z1)
            gradtest(net, x1, z1)
        end
    end
end
            

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


