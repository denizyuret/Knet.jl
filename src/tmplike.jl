# GPU memory allocation is very expensive.  So we create an
# application specific memory manager.  Typically same type, size, and
# number of arrays are needed in every iteration of training and
# testing.  During training these arrays should not be overwritten
# until the backward pass.  During testing they should be recycled as
# much as possible.

# We will have a Dict((arrayType,arrayLength)=>arrays):

!isdefined(:TmpDict) && (TmpDict = Dict())

# Each value in the dict will hold arrays of same type and length with
# idx pointing to the last one used:

type TmpStack; arr::Vector; idx::Int; TmpStack()=new([],0); end

# When we are done with an iteration, we reset idx=0 instead of
# freeing the arrays so we can reuse them:

tmpfree()=(for s in values(TmpDict); s.idx=0; end)

# This is the main function, to be used like "similar":

function tmplike(a, dims::Dims=size(a))
    s = get!(TmpStack, TmpDict, (typeof(a),prod(dims)))
    s.idx += 1
    if s.idx > length(s.arr)
        push!(s.arr, similar(a,dims))
        s.idx > length(s.arr) && error("short stack")
    end
    if size(s.arr[s.idx]) != dims
        s.arr[s.idx] = reshape(s.arr[s.idx], dims)
    end
    return s.arr[s.idx]
end

tmplike(a, dims::Int...)=tmplike(a, dims)

function gpuinfo()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    nbytes=convert(Int,mfree[1])
    narray=length(CUDArt.cuda_ptrs)
    println((:free,nbytes,:cuda_ptrs,narray))
    for (t,s) in TmpDict
        println((t...,length(s.arr)))
    end
end
