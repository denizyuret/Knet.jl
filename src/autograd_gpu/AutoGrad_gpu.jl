module AutoGrad_gpu

import AutoGrad: addto!, addtoindex!, matches, ungetindex, Sparse, zeroslike
import LinearAlgebra: axpy!
import Base: +, -, copyto!
import CUDA, Knet, AutoGrad
using CUDA: CuArray, CuPtr, functional
using Knet.KnetArrays: DevArray, KnetArray, Cptr, cuallocator
using Knet.LibKnet8: @knet8
using AutoGrad: AutoGrad, Sparse, recording, Result, Node, Tape, Value, Arg, value, set_gc_function
using Base.Broadcast: Broadcasted

include("addto.jl")
include("convert.jl")
include("cuarrays.jl")
include("getindex.jl")
include("sparse.jl")

include("priorityqueue.jl") # using DataStructures: DataStructures, PriorityQueue, dequeue!, dequeue_pair! # peek
include("gcnode.jl")
include("gcnode_kptr.jl")

end
