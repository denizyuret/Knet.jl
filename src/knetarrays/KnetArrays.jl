module KnetArrays

include("kptr.jl");   export KnetPtr, Cptr, gc, knetgc
include("karray.jl"); export KnetArray, KnetMatrix, KnetVector, KnetVecOrMat, DevArray, ka

include("getindex.jl")
include("abstractarray.jl")
include("broadcast.jl")
include("cat.jl")
include("comparison.jl")
include("copy.jl")
include("deepcopy.jl"); export cpucopy, gpucopy
include("dotview.jl")
include("linalg.jl")
include("random.jl"); export setseed, seed!
include("reshape.jl")
include("show.jl")
include("statistics.jl")

include("binary.jl")
include("unary.jl")
include("reduction.jl")

include("serialization.jl") # serialize and deserialize of KnetArrays
include("jld2.jl"); export save, load, @save, @load # deprecated, use FileIO and JLD2

end
