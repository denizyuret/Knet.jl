using Base.Test
using CUDArt
using KUnet
using CUDArt: ContiguousArray
using Base.Test: Success, Failure, Error
import Base.Test: default_handler

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

function gradcheck(x, dx, lossfn; iter=10, 
                   epsilon=cbrt(eps(eltype(x))), 
                   delta=(eltype(x)==Float64) ? 1e-4 : 1e-3)
    maxdiff = 0.0
    for i=1:iter
        r = rand(1:length(x))
        xr0 = x[r]
        # Do not cross 0 for softloss
        xr1 = xr0 - delta; xr0>0 && xr1<0 && (xr1=0)
        xr2 = xr0 + delta; xr0<0 && xr2>0 && (xr2=0)
        x[r] = xr1; loss1 = lossfn()
        x[r] = xr2; loss2 = lossfn()
        x[r] = xr0
        dxr = (loss2 - loss1) / (xr2 - xr1)
        dx[r]
        absdiff = abs(dxr - dx[r])/(abs(dxr) + abs(dx[r]))
        absdiff > maxdiff && (maxdiff = absdiff)
    end
    # @show (maxdiff, epsilon, delta)
    return maxdiff < epsilon
end

CudaLayer(l)=(isa(l, Mmul) ? Mmul(l.w.data) :
              isa(l, Conv) ? Conv(l.w.data) :
              isa(l, Bias) ? Bias(l.b.data) : 
              isa(l, Drop) ? Drop(l.dropout) :
              isa(l, Pool) ? Pool(l.pd) :
              typeof(l)())

# Test each layer for: 1D-5D, gpu/cpu, float32/float64

ninst = 64
KUnet.atype(Array)
ft = dims = x1 = x2 = y1 = y2 = z1 = z2 = dx1 = dx2 = dy1 = dy2 = l1 = l2 = qloss = xdrop = nothing
lossfn1 = lossfn2 = lossfn3 = nothing

for ft in (Float32,Float64)
# for ft in (Float64,)
    KUnet.ftype(ft)
    gradeps = cbrt(eps(ft))
    for dims in 1:5
#     for dims in 3:5
        x1 = ((dims == 1) ? rand(ft, 784) :
              (dims == 2) ? rand(ft, 784, ninst) :
              (dims == 3) ? rand(ft, 28, 28, ninst) :
              (dims == 4) ? rand(ft, 28, 14, 2, ninst) :
              (dims == 5) ? rand(ft, 14, 7, 4, 2, ninst) :
              error("dims=$dims."))
        xdrop = rand(ft, size(x1))
        xchan = (dims==1 ? length(x1) : size(x1, dims-1))
        xinst = (dims==1 ? 1 : size(x1, dims))
        xrows = div(length(x1), xinst)
        for l1 in (
                   # Bias(Param(rand(ft, xchan))),
                   # Drop(0.5),
                   # Logp(), 
                   # Mmul(10,xrows),
                   # Relu(), 
                   # Sigm(), 
                   # Soft(),
                   # Tanh(),
                   )
            # in(typeof(l1), (Conv, Pool)) && dims != 4 && continue
            # in(typeof(l1), (Drop, Logp)) && ft == Float64 && continue
            @show (ft, dims, size(x1), typeof(l1))

            qloss1 = QuadLoss()
            y1 = forw(l1, copy(x1); xdrop=xdrop)
            forw(qloss1, y1)  # network output
            z1 = rand(ft, size(y1))  # desired answers
            loss1 = loss(qloss1, z1)  # loss value
            dy1 = back(qloss1, copy(z1))
            dx1 = back(l1, copy(dy1))

            if KUnet.GPU
                KUnet.atype(CudaArray)
                qloss2 = QuadLoss()
                l2 = CudaLayer(l1)
                x2 = CudaArray(x1)
                y2 = forw(l2, copy(x2); xdrop=xdrop)
                @test isapprox(y1, y2)
                forw(qloss2, y2)
                z2 = CudaArray(z1)
                loss2 = loss(qloss2, z2)
                @show (loss1, loss2)
                @test isapprox(loss1, loss2)
                dy2 = back(qloss2, copy(z2))
                dx2 = back(l2, copy(dy2))
                @test isapprox(dx1, dx2)
                KUnet.atype(Array)
            end

            # Gradient check:
            lossfn1=()->loss(qloss1,z1)
            @test gradcheck(y1, dy1, lossfn1)
            lossfn2=()->(forw(qloss1,forw(l1,copy(x1);xdrop=xdrop));loss(qloss1,z1))
            @test gradcheck(x1, dx1, lossfn2)
            w1 = nothing
            for n in names(l1); isa(l1.(n), Param) && (w1=l1.(n)); end
            if w1 != nothing
                @test gradcheck(w1.data, w1.diff, lossfn2)
            end  # if w1 != nothing
        end # for l1

        for l1 in (
                   LogpLoss(),
                   QuadLoss(),
                   SoftLoss(),
                   XentLoss(),
                   )
            if isa(l1, QuadLoss)
                y1 = x1
                z1 = rand(ft, size(y1))
                @show (ft, dims, size(y1), typeof(l1))
            else
                dims > 2 && continue  # cross entropy losses are accurate for a few classes
                y1 = (dims==1 ? rand(ft, 10) : rand(ft, 10, size(x1,dims)))
                z1 = forw(Soft(), 5*rand(ft, size(y1)))
                @show (ft, dims, size(y1), typeof(l1))
                nxones = ones(ft, (dims==1 ? 1 : size(x1, dims)))
                @test isapprox(nxones, vec(sum(z1,1)))
                if isa(l1, LogpLoss) 
                    forw(Logp(), y1)
                    @test isapprox(nxones, vec(sum(exp(y1),1)))
                end
                if isa(l1, SoftLoss)
                    forw(Soft(), y1)
                    @test isapprox(nxones, vec(sum(y1,1)))
                end
            end

            forw(l1, y1)
            dy1 = back(l1, copy(z1))
            
            if KUnet.GPU
                KUnet.atype(CudaArray)
                l2 = CudaLayer(l1)
                y2 = CudaArray(y1)
                z2 = CudaArray(z1)
                forw(l2,y2)
                dy2 = back(l2, copy(z2))
                @test isapprox(dy1,dy2)
                KUnet.atype(Array)
            end

            lossfn3=()->(isa(l1,LogpLoss) ? forw(l1,forw(Logp(),copy(y1))) :
                         isa(l1,SoftLoss) ? forw(l1,copy(y1)./sum(y1,1)) :
                         forw(l1,y1); loss(l1,z1))
            @test gradcheck(y1, dy1, lossfn3)

        end # for l1 in lossfns

        for l1 in (             # no CPU impl or gradcheck
                   # Conv(5,5,xchan,10), # only gpu, only 4D
                   # Pool(2, max(1,dims-2)), # only gpu, only 4D
                   )
            dims != 4 && continue
            @show (ft, dims, size(x1), typeof(l1))
            KUnet.atype(CudaArray)
            qloss2 = QuadLoss()
            l2 = CudaLayer(l1)
            x2 = CudaArray(x1)
            y2 = forw(l2, copy(x2); xdrop=xdrop); @test y2!=nothing
            # @test isapprox(y1, y2)
            y3 = forw(qloss2, y2); @test y3!=nothing
            # z2 = CudaArray(z1)
            z2 = CudaArray(rand(ft, size(y2)))
            @show loss2 = loss(qloss2, z2)
            # @show (loss1, loss2)
            # @test isapprox(loss1, loss2)
            dy2 = back(qloss2, copy(z2)); @test dy2!=nothing
            dx2 = back(l2, copy(dy2)); @test dx2!=nothing
            # @test isapprox(dx1, dx2)
            KUnet.atype(Array)
        end # for l1 in conv,pool

    end # for dims
end # for ft
