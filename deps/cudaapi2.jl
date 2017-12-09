module CUDAapi2
using Compat
include("logging.jl")
include("compatibility.jl")
include("discovery.jl")
export find_library, libnvml
end
