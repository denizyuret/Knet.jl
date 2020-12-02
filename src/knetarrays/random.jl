export setseed, seed!
import Random: rand!, randn!, Random
using Knet.KnetArrays: KnetArray

rand!(a::KnetArray)=(rand!(CuArray(a)); a)
randn!(a::KnetArray)=(randn!(CuArray(a)); a)

function setseed(x)
    @warn "setseed() is deprecated, please use Random.seed!() and/or CUDA.seed!() instead" maxlog=1
    CUDA.functional() && CUDA.seed!(x)
    Random.seed!(x) 
end

"Call both CUDA.seed! (if available) and Random.seed!"
function seed!(x)
     CUDA.functional() && CUDA.seed!(x)
     Random.seed!(x)
end

