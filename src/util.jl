# These functions are not exported, use with `Knet.func()`:

"`Knet.atype()` gives the current default array type: by default `KnetArray{Float32}` if `gpu() >= 0`, `Array{Float32}` otherwise. The user can change the default array type using e.g. Knet.atype()=CuArray{Float32}. `Knet.atype(x)` converts `x` to `atype()`."
atype()=(gpu() >= 0 ? KnetArray{Float32} : Array{Float32})
atype(x)=convert(atype(),x)

"`Knet.training()` returns `true` only inside a `@diff` context, e.g. during a training iteration of a model. This is used by dropout, batchnorm etc to have code run differently during training vs inference."
training() = AutoGrad.recording()

"`Knet.seed!(n)` sets the seed for both the CPU and the GPU random number generators for replicability."
seed!(n::Integer)=(CUDA.functional() && CUDA.seed!(n); Random.seed!(n))

"`Knet.dir(path...) constructs a path relative to Knet root, e.g. `Knet.dir(\"src\",\"Knet.jl\")` => `/home/dyuret/.julia/dev/Knet/src/Knet.jl`"
dir(path...) = joinpath(dirname(@__DIR__),path...)

