# This file contains various utilities, compatibility fixes and hacks.
# Hopefully it will shrink down to nothing as things get fixed in the
# original packages.

if GPU   ########## CUDA extensions:

const libkunet = find_library(["libkunet"], [Pkg.dir("KUnet/src")])

Base.convert{T,S}(::Type{CudaArray{T}}, x::Array{S})=CudaArray(convert(Array{T}, x))
Base.reshape(a::CudaArray, dims::Dims)=reinterpret(eltype(a), a, dims)
Base.reshape(a::CudaArray, dims::Int...)=reshape(a, dims)

# TODO: Float64 support
typealias Cmat Ptr{Float32}

# arrays.jl:297, need these so generic code works with SubArrays:
Base.copy!{T}(dst::AbstractCudaArray{T}, dstI::(Union(Int,Range1{Int})...), src::SubArray{T}, srcI::(Union(Int,Range1{Int})...))=(s=sub(src, srcI...); copy!(dst, dstI, s.parent, s.indexes))
Base.copy!{T}(dst::SubArray{T}, dstI::(Union(Int,Range1{Int})...), src::AbstractCudaArray{T}, srcI::(Union(Int,Range1{Int})...))=(d=sub(dst, dstI...); copy!(d.parent, d.indexes, src, srcI))

# TODO: these don't hang high enough in the type hierarchy
# TODO: non of these implementations are complete, they are just barely sufficient to make kunet work.
using InplaceOps: Transpose
InplaceOps.op_ctranspose(x::CudaVecOrMat)=Transpose(x)
InplaceOps.mul!(O::CudaVecOrMat, A::CudaVecOrMat, B::CudaVecOrMat) = CUBLAS.gemm!('N','N',one(eltype(O)),A,B,zero(eltype(O)),O)  # InplaceOps.jl:53
InplaceOps.mul!(O::CudaVecOrMat, A::Transpose, B::CudaVecOrMat) = CUBLAS.gemm!('T','N',one(eltype(O)),A.obj,B,zero(eltype(O)),O)
InplaceOps.mul!(O::CudaVecOrMat, A::CudaVecOrMat, B::Transpose) = CUBLAS.gemm!('N','T',one(eltype(O)),A,B.obj,zero(eltype(O)),O)
# TODO: generalize to N-D:
InplaceOps.badd!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, B::CudaVecOrMat) = (ccall((:badd,libkunet),Void,(Cint,Cint,Cmat,Cmat),size(A,1),size(A,2),A,B);A) # InplaceOps.jl:83
InplaceOps.bmul!(::Type{InplaceOps.Inplace{1}}, A::CudaArray, x::Number) = CUBLAS.scal!(length(A), float32(x), A, 1)
InplaceOps.bsub!(::Type{InplaceOps.Inplace{1}}, A::CudaArray, B::CudaArray) = CUBLAS.axpy!(length(A), -1.0f0, B, 1, A, 1)
InplaceOps.bsub!(::Type{InplaceOps.Inplace{1}}, A::CudaArray, x::Number) = (ccall((:add1,libkunet),Void,(Cint,Cfloat,Cmat),length(A),-x,A);A)

# # I could not get this to work:
# import Base: convert, promote_rule
# convert(::Type{Mat},x::Transpose{Mat})=x.obj
# promote_rule(::Type{Mat},::Type{Transpose{Mat}})=Mat

# TODO: add error checking here since this is not a full implementation of sum!
Base.sum!(r::CudaVecOrMat, A::CudaMatrix) = ccall((:bsum,libkunet),Void,(Cint,Cint,Cmat,Cmat),size(A,1),size(A,2),A,r) # reducedim.jl:226
Base.zeros(A::CudaArray)=CUBLAS.scal!(length(A), zero(eltype(A)), copy(A), 1)
Base.rand!(A::CudaArray{Float32})=(ccall((:randfill32,libkunet),Void,(Cint,Ptr{Float32}),length(A),A); A)
Base.rand!(A::CudaArray{Float64})=(ccall((:randfill64,libkunet),Void,(Cint,Ptr{Float64}),length(A),A); A)
Base.fill!(A::CudaArray,x::Number)=(ccall((:fill,libkunet),Void,(Cint,Cfloat,Cmat),length(A),x,A); A)

# For debugging
function gpumem()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart.so"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    convert(Int,mfree[1])
end

using CUDArt: ContiguousArray

# CUDArt copy! doesn't work for 4D arrays, so let's make this Base interface work:
function Base.copy!{T}(dst::ContiguousArray{T}, di::Integer, src::ContiguousArray{T}, si::Integer, n::Integer; stream=null_stream)
    if si+n-1 > length(src) || di+n-1 > length(dst) || di < 1 || si < 1
        throw(BoundsError())
    end
    nbytes = n * sizeof(T)
    dptr = pointer(dst) + (di-1) * sizeof(T)
    sptr = pointer(src) + (si-1) * sizeof(T)
    CUDArt.rt.cudaMemcpyAsync(dptr, sptr, nbytes, CUDArt.cudamemcpykind(dst, src), stream)
    return dst
end

end	########## CUDA extensions


# our version of srand sets both gpu and cpu
function srandom(n::Integer)
    srand(n)
    GPU && ccall((:gpuseed,libkunet),Void,(Culonglong,),convert(Culonglong, n))
end

function chksize(l, n, a, dims=size(a); fill=nothing)
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

# TODO: generalize this to N-D
function shufflexy!(x, y)
    xrows,xcols = size(x)
    yrows,ycols = size(y)
    @assert xcols == ycols
    x1 = Array(eltype(x), xrows)
    y1 = Array(eltype(y), yrows)
    for n = xcols:-1:2
        r = rand(1:n)
        r == n && continue
        nx = (n-1)*xrows+1; ny = (n-1)*yrows+1
        rx = (r-1)*xrows+1; ry = (r-1)*yrows+1
        copy!(x1, 1, x, nx, xrows)
        copy!(y1, 1, y, ny, yrows)
        copy!(x, nx, x, rx, xrows)
        copy!(y, ny, y, ry, yrows)
        copy!(x, rx, x1, 1, xrows)
        copy!(y, ry, y1, 1, yrows)
    end
end

# arrays.jl:297, need these so generic code works with cpu arrays:
Base.copy!{T}(dst::AbstractArray{T}, dstI::(Union(Int,Range1{Int})...), src::AbstractArray{T}, srcI::(Union(Int,Range1{Int})...))=copy!(sub(dst, dstI...), sub(src, srcI...))

# when gc works these should not be necessary:
if isdefined(:CUDArt)
    import CUDArt: free, to_host
end
free(x)=x
to_host(x)=x

size2(y)=(nd=ndims(y); (nd==1 ? (length(y),1) : (stride(y, nd), size(y, nd))))
issimilar(a,b)=((typeof(a)==typeof(b)) && (size(a)==size(b)))
accuracy(y,z)=mean(findmax(y,1)[2] .== findmax(z,1)[2])


# TODO: We should leave these up to the layers:
# istransient(l,n)=(isa(l,Layer) && in(n,(:x,:y,:z,:dx,:dy,:dz,:xdrop)))  # no need to copy or save these
# # clean(l::Layer)=(for f in names(l); isdefined(l,f) && istransient(l,f) && (l.(f)=similar(l.(f),(0,0))); end)
# # clean(n::Net)=(for l in n; clean(l); end)

# # # We should deprecate this function, now that we have more than one type of layer.
# # # Julia v0.4 allows Net as a constructor name, but v0.3 does not:
# # # Net(f::Function, d::Integer...; o...) = (n=Layer[]; for i=2:length(d); push!(n, (i<length(d)) ? Layer(f,d[i-1],d[i];o...) : Layer(d[i-1],d[i];o...)); end; n)
# # newnet(f::Function, d::Integer...; o...) = (n=Layer[]; for i=2:length(d); push!(n, (i<length(d)) ? Layer(f,d[i-1],d[i];o...) : Layer(d[i-1],d[i];o...)); end; n)

# function Base.copy(l::Union(Layer,Param), to=nothing)
#     ll = typeof(l)()
#     for n in fieldnames(l)
#         isdefined(l,n) || continue
#         istransient(l,n) && continue
#         iscnull(l.(n)) && continue
#         isa(l.(n), AbstractArray) && isempty(l.(n)) && continue
#         if (to == :test)   # minimum needed for predict
#             n == :w && (ll.(n) = to_host(l.(n)))
#             n == :b && (ll.(n) = to_host(l.(n)))
#             n == :f && (ll.(n) = l.(n))
#         elseif ((to == :cpu) && isdefined(:CUDArt) && isa(l.(n), CudaArray))
#             ll.(n) = to_host(l.(n))
#         elseif ((to == :gpu) && isdefined(:CUDArt) && isa(l.(n), AbstractArray))
#             ll.(n) = CudaArray(l.(n))
#         elseif (isa(l.(n), Param))
#             ll.(n) = copy(l.(n), to)
#         else
#             ll.(n) = copy(l.(n))
#         end
#     end
#     return ll
# end

# Base.copy(net::Net, to=nothing)=map(layer->copy(layer,to), net)
# iscnull(x)=(in(:ptr,names(x)) && (C_NULL==convert(typeof(C_NULL), x.ptr)))


