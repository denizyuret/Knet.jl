module KnetArrays

include("kptr.jl"); export KnetPtr, Cptr, gc
include("karray.jl"); export KnetArray, KnetMatrix, KnetVector, KnetVecOrMat, DevArray

include("getindex.jl")
include("abstractarray.jl")
include("broadcast.jl")
include("cat.jl")
include("comparison.jl")
include("copy.jl")
include("dotview.jl")
include("linalg.jl")
include("random.jl")
include("reshape.jl")
include("show.jl")
include("statistics.jl")

include("binary.jl")
include("unary.jl")
include("reduction.jl")

end
