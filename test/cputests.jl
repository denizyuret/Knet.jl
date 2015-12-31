using Knet, Base.Test

# Uncomment these if you want lots of messages:
# import Base.Test: default_handler, Success, Failure, Error
# default_handler(r::Success) = info("$(r.expr)")
# default_handler(r::Failure) = warn("FAIL: $(r.expr)")
# default_handler(r::Error)   = warn("$(r.err): $(r.expr)")

load_only = true
isapprox3(a,b,c)=all(map((x,y,z)->isapprox(x,y;rtol=z), a,b,c))

include(Pkg.dir("Knet/examples/linreg.jl"))
#@test LinReg.main("") == (0.0005497846637255735,32.77257400591496, 0.11265426200775067) #gpu
@test isapprox3(LinReg.main(""), (0.0005439055920768844,32.772149551563935,0.11210599283551381), (1e-4,1e-4,1e-4))

include(Pkg.dir("Knet/examples/mnist2d.jl"))
#@test MNIST2D.main("--epochs 1") == (0.3204898f0, 32.93997f0, 4.614684f0) #gpu
@test isapprox3(MNIST2D.main("--epochs 1"), (0.30533937f0,33.110886f0,5.677294f0), (1e-4,1e-4,1e-4))

#@test MNIST2D.main("--epochs 1 --ysparse") == (0.3204898f0, 32.93997f0, 4.614684f0) #gpu
@test isapprox3(MNIST2D.main("--epochs 1 --ysparse"), (0.30533937f0,33.110886f0,5.677294f0), (1e-4,1e-4,1e-4))

warn("Need to implement: A_mul_B!(::Array{Float32,2}, ::Array{Float32,2}, ::SparseMatrixCSC{Float32,Int32}); the other direction is implemented in sparse/linalg.jl")
# @test isapprox3(MNIST2D.main("--epochs 1 --xsparse"), (0.32048982f0,32.93997f0,4.6146836f0), (0.01,0.001,0.1))
# @test isapprox3(MNIST2D.main("--epochs 1 --xsparse --ysparse"), (0.3204898f0,32.93997f0,4.614684f0), (0.01,0.001,0.1))

warn("cpu conv not implemented yet")
# include(Pkg.dir("Knet/examples/mnist4d.jl"))
# @test isapprox3(MNIST4D.main("--epochs 1"), (0.20520441331629022,65.88371276855469,114.9615707397461), (0.02,0.001,0.2))

# Differences below due mostly to random initialization

include(Pkg.dir("Knet/examples/adding.jl"))
#@test Adding.main("--epochs 1") == (0.22713459f0,3.3565507f0,5.3267756f0) #gpu
@test isapprox3(Adding.main("--epochs 1"), (0.21368636f0,3.4062932f0,6.621738f0), (1e-4,1e-4,1e-4))

#@test Adding.main("--epochs 1 --nettype lstm") == (0.24768005f0,3.601481f0,1.2290705f0) #gpu
@test isapprox3(Adding.main("--epochs 1 --nettype lstm"), (0.24432828f0,3.6435652f0,1.1989625f0), (1e-4,1e-4,1e-4))

seqdata = Pkg.dir("Knet/data/seqdata.txt")

include(Pkg.dir("Knet/examples/rnnlm.jl"))
#@test RNNLM.main("--max_max_epoch 1 --dense seqdata.txt") == (30.65637197763184, 110.81809997558594,29.36899185180664) #gpu
@test isapprox3(RNNLM.main("--max_max_epoch 1 --dense $seqdata"), (30.349935446042096,110.69515228271484,26.056880950927734), (1e-4,1e-4,1e-4))

include(Pkg.dir("Knet/examples/copyseq.jl"))
#@test CopySeq.main("--epochs 1 --dense seqdata.txt") == (40.00286169097269, 30.352935791015625,1.646486520767212)
@test isapprox3(CopySeq.main("--epochs 1 --dense $seqdata"), (40.00211618458305,29.903160095214844,1.4534363746643066), (1e-4,1e-4,1e-4))

### DEAD CODE:
# using Compat
# using Knet
# @date include("testdense.jl")
# @date include("testsparse.jl")
# @date include("testconvert.jl")
# @date include("testlinalg.jl")
# @date include("testcolops.jl")
# @date include("testmnist.jl")
# @date include("testlayers.jl")
# @date include("testperceptron.jl")
# @date include("testkperceptron.jl")
# # @date include("tutorial.jl")
