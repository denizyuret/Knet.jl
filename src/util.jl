# This file contains various utilities, compatibility fixes and hacks.
# Hopefully it will shrink down to nothing as things get fixed in the
# original packages.

function similar!(l, n, a, dims=size(a); fill=nothing)
    if !isdefined(l,n) 
        l.(n) = similar(a, dims)
        fill != nothing && fill!(l.(n), fill)
    elseif size(l.(n)) != dims
        free(l.(n))
        l.(n) = similar(a, dims)
        fill != nothing && fill!(l.(n), fill)
    end
    return l.(n)
end

issimilar(a,b)=((typeof(a)==typeof(b)) && (size(a)==size(b)))
size2(y)=(nd=ndims(y); (nd==1 ? (length(y),1) : (stride(y, nd), size(y, nd))))
accuracy(y,z)=mean(findmax(y,1)[2] .== findmax(z,1)[2])

if GPU   ########## CUDA extensions:

const libkunet = find_library(["libkunet"], [Pkg.dir("KUnet/src")])

convert{T,S}(::Type{CudaArray{T}}, x::Array{S})=CudaArray(convert(Array{T}, x))
reshape(a::CudaArray, dims::Dims)=reinterpret(eltype(a), a, dims)
reshape(a::CudaArray, dims::Int...)=reshape(a, dims)
rand!(A::CudaArray{Float32})=(ccall((:randfill32,libkunet),Void,(Cint,Ptr{Float32}),length(A),A); A)
rand!(A::CudaArray{Float64})=(ccall((:randfill64,libkunet),Void,(Cint,Ptr{Float64}),length(A),A); A)
fill!(A::CudaArray,x::Number)=cudnnSetTensor(A, x)

using CUDArt: ContiguousArray
function copy!{T}(dst::ContiguousArray{T}, di::Integer, src::ContiguousArray{T}, si::Integer, n::Integer; stream=null_stream)
    if si+n-1 > length(src) || di+n-1 > length(dst) || di < 1 || si < 1
        throw(BoundsError())
    end
    nbytes = n * sizeof(T)
    dptr = pointer(dst) + (di-1) * sizeof(T)
    sptr = pointer(src) + (si-1) * sizeof(T)
    CUDArt.rt.cudaMemcpyAsync(dptr, sptr, nbytes, CUDArt.cudamemcpykind(dst, src), stream)
    return dst
end

# For debugging
function gpumem()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart.so"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    convert(Int,mfree[1])
end

# our version of srand sets both gpu and cpu
function gpuseed(n::Integer)
    srand(n)
    GPU && ccall((:gpuseed,libkunet),Void,(Culonglong,),convert(Culonglong, n))
end

import CUDArt: to_host

end	########## CUDA extensions

# Need this so code works without gpu
to_host(x)=x
