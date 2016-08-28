tmplike(a...)=similar(a...)
tmpfree()=nothing

function gpuinfo()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    nbytes=convert(Int,mfree[1])
    narray=length(CUDArt.cuda_ptrs)
    println((:free,nbytes,:cuda_ptrs,narray))
end
