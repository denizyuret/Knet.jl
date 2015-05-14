using CUDArt
using KUnet
using Base.Test
require("mnist.jl")

# Uncomment this if you want lots of messages:
Base.Test.default_handler(r::Base.Test.Success) = info("$(r.expr)")
Base.Test.default_handler(r::Base.Test.Failure) = warn("FAIL: $(r.expr)")
Base.Test.default_handler(r::Base.Test.Error)   = warn("ERROR: $(r.expr)")

epseq(x,y)=(maximum(abs(to_host(x)-to_host(y))) < 10*eps(eltype(x)))

CudaLayer(l)=(isa(l, Mmul) ? Mmul(l.w.data) :
              isa(l, Conv) ? Conv(l.w.data) :
              isa(l, Bias) ? Bias(l.b.data) : 
              isa(l, Drop) ? Drop(l.dropout) :
              isa(l, Pool) ? Pool(first(l.pd.dims)) :
              typeof(l)())

# Test each layer for: 1D-5D, gpu/cpu, float32/float64

ninst = 64
KUnet.atype(Array)
x1 = x2 = y1 = y2 = dx1 = dx2 = dy1 = dy2 = l1 = l2 = nothing

for ft in (Float32,Float64) # TODO: add Float64
    KUnet.ftype(ft)
    for dims in 1:5
        if dims == 1
            x1 = convert(Array{ft}, squeeze(MNIST.xtst[:,1], 2))
        else
            x1 = convert(Array{ft}, MNIST.xtst[:,1:ninst])
            x1 = ((dims == 2) ? x1 :
                  (dims == 3) ? reshape(x1, 28, 28, ninst) :
                  (dims == 4) ? reshape(x1, 28, 14, 2, ninst) :
                  (dims == 5) ? reshape(x1, 14, 7, 4, 2, ninst) :
                  error("dims=$dims."))
        end
        xdrop = rand(ft, size(x1))
        xchan = (dims==1 ? length(x1) : size(x1, dims-1))
        xinst = (dims==1 ? 1 : size(x1, dims))
        xrows = div(length(x1), xinst)
        for l1 in (
                   Bias(Param(rand(ft, xchan))),
                   Conv(5,5,xchan,10),
                   Drop(0.5),
                   Logp(), 
                   Mmul(10,xrows),
                   Pool(2),
                   Relu(), 
                   Sigm(), 
                   Tanh(),
                   )
            in(typeof(l1), (Conv, Pool)) && dims != 4 && continue
            @show (ft, dims, size(x1), typeof(l1))
            y1 = copy(x1)
            @test (y1 = forw(l1, y1; xdrop=xdrop))!=nothing
            dy1 = rand(ft, size(y1))
            dx1 = copy(dy1)
            @test (dx1 = back(l1, dx1))!=nothing
            if KUnet.GPU
                KUnet.atype(CudaArray)
                l2 = CudaLayer(l1)
                y2 = CudaArray(x1)
                @test (y2 = forw(l2, y2; xdrop=xdrop))!=nothing
                dy2 = (size(dy1)==size(y2) ? CudaArray(dy1) : CudaArray(rand(ft, size(y2))))
                dx2 = copy(dy2)
                @test (dx2 = back(l2, dx2))!=nothing
                @test epseq(y1, y2)
                @test epseq(dx1, dx2)
                KUnet.atype(Array)
            end
        end
    end
end
