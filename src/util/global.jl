# export atype, training, seed!, dir, cuallocator ## do not export, use with Knet.x
import AutoGrad                 # recording
import CUDA                     # functional, seed!
import Random                   # seed!

# TODO: deprecate atype(), promote array_type[]
"`Knet.atype()` gives the current default array type: by default `KnetArray{Float32}` if `gpu() >= 0`, `Array{Float32}` otherwise. The user can change the default array type using e.g. Knet.atype()=CuArray{Float32}. `Knet.atype(x)` converts `x` to `atype()`."
atype()=array_type[]
atype(x)=convert(atype(),x)
const array_type = Ref{Type}(Array{Float32})

"`Knet.training()` returns `true` only inside a `@diff` context, e.g. during a training iteration of a model. This is used by dropout, batchnorm etc to have code run differently during training vs inference."
training() = AutoGrad.recording()

"`Knet.seed!(n)` sets the seed for both the CPU and the GPU random number generators for replicability."
seed!(n::Integer)=(CUDA.functional() && CUDA.seed!(n); Random.seed!(n))

"`Knet.dir(path...) constructs a path relative to Knet root, e.g. `Knet.dir(\"src\",\"Knet.jl\")` => `/home/dyuret/.julia/dev/Knet/src/Knet.jl`"
dir(path...) = joinpath(dirname(dirname(@__DIR__)),path...)

"`Knet.cuallocator[]` is `true` by default to use the CUDA.jl allocator, set `Knet.cuallocator[]=false` to use the Knet allocator."
cuallocator

function gpu(x...)
    @warn "gpu() is deprecated, please use CUDA.device instead" maxlog=1
    CUDA.functional() ? CUDA.device().handle : -1
end

# TODO: deprecate this
# To see debug output, start julia with `JULIA_DEBUG=Knet julia`
# To perform profiling, set ENV["KNET_TIMER"] to "true" and rebuild Knet. (moved this to gpu.jl)
# The @dbg macro below evaluates `ex` only when debugging. The @debug macro prints stuff as documented in Julia.
#macro dbg(ex); :(if Base.CoreLogging.current_logger_for_env(Base.CoreLogging.Debug,:none,Knet)!==nothing; $(esc(ex)); end); end
