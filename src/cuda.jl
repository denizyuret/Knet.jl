# This file contains various compatibility fixes and hacks.  Hopefully
# it will shrink down to nothing as things get fixed in the original
# packages.

# arrays.jl:297, need these so generic code works with cpu arrays:
import Base: copy!
copy!{T}(dst::AbstractArray{T}, dstI::(Union(Int,Range1{Int})...), src::AbstractArray{T}, srcI::(Union(Int,Range1{Int})...))=copy!(sub(dst, dstI...), sub(src, srcI...))

# when gc works these should not be necessary:
if isdefined(:CUDArt)
    import CUDArt: free, to_host
end
free(x)=x
to_host(x)=x

istransient(l,n)=(isa(l,Layer) && in(n,(:y,:x,:dx,:xdrop)))  # no need to copy or save these
clean(l::Layer)=(for f in names(l); isdefined(l,f) && istransient(l,f) && (l.(f)=similar(l.(f),(0,0))); end)
clean(n::Net)=(for l in n; clean(l); end)

# Julia v0.4 allows Net as a constructor name, but v0.3 does not:
# Net(f::Function, d::Integer...; o...) = (n=Layer[]; for i=2:length(d); push!(n, (i<length(d)) ? Layer(f,d[i-1],d[i];o...) : Layer(d[i-1],d[i];o...)); end; n)
newnet(f::Function, d::Integer...; o...) = (n=Layer[]; for i=2:length(d); push!(n, (i<length(d)) ? Layer(f,d[i-1],d[i];o...) : Layer(d[i-1],d[i];o...)); end; n)
# We should deprecate this function, now that we have more than one type of layer.

import Base.copy

function copy(l::Union(Layer,UpdateParam), to=nothing)
    ll = typeof(l)()
    for n in fieldnames(l)
        isdefined(l,n) || continue
        istransient(l,n) && continue
        iscnull(l.(n)) && continue
        isa(l.(n), AbstractArray) && isempty(l.(n)) && continue
        if (to == :test)   # minimum needed for predict
            n == :w && (ll.(n) = to_host(l.(n)))
            n == :b && (ll.(n) = to_host(l.(n)))
            n == :f && (ll.(n) = l.(n))
        elseif ((to == :cpu) && isdefined(:CUDArt) && isa(l.(n), CudaArray))
            ll.(n) = to_host(l.(n))
        elseif ((to == :gpu) && isdefined(:CUDArt) && isa(l.(n), AbstractArray))
            ll.(n) = CudaArray(l.(n))
        elseif (isa(l.(n), UpdateParam))
            ll.(n) = copy(l.(n), to)
        else
            ll.(n) = copy(l.(n))
        end
    end
    return ll
end

copy(net::Net, to=nothing)=map(layer->copy(layer,to), net)
iscnull(x)=(in(:ptr,names(x)) && (C_NULL==convert(typeof(C_NULL), x.ptr)))


if isdefined(:CUDArt)   ########## CUDA extensions:

typealias Cmat Ptr{Float32}

# arrays.jl:297, need these so generic code works with SubArrays:
copy!{T}(dst::AbstractCudaArray{T}, dstI::(Union(Int,Range1{Int})...), src::SubArray{T}, srcI::(Union(Int,Range1{Int})...))=(s=sub(src, srcI...); copy!(dst, dstI, s.parent, s.indexes))
copy!{T}(dst::SubArray{T}, dstI::(Union(Int,Range1{Int})...), src::AbstractCudaArray{T}, srcI::(Union(Int,Range1{Int})...))=(d=sub(dst, dstI...); copy!(d.parent, d.indexes, src, srcI))

# TODO: these don't hang high enough in the type hierarchy
# TODO: non of these implementations are complete, they are just barely sufficient to make kunet work.
import InplaceOps: op_ctranspose, Transpose, mul!, badd!, bmul!, bsub! 
op_ctranspose(x::CudaVecOrMat)=Transpose(x)
mul!(O::CudaVecOrMat, A::CudaVecOrMat, B::CudaVecOrMat) = CUBLAS.gemm!('N','N',one(eltype(O)),A,B,zero(eltype(O)),O)  # InplaceOps.jl:53
mul!(O::CudaVecOrMat, A::Transpose, B::CudaVecOrMat) = CUBLAS.gemm!('T','N',one(eltype(O)),A.obj,B,zero(eltype(O)),O)
mul!(O::CudaVecOrMat, A::CudaVecOrMat, B::Transpose) = CUBLAS.gemm!('N','T',one(eltype(O)),A,B.obj,zero(eltype(O)),O)
badd!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, B::CudaVecOrMat) = (ccall((:badd,libkunet),Void,(Cint,Cint,Cmat,Cmat),size(A,1),size(A,2),A,B);A) # InplaceOps.jl:83
bmul!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, x::Float32) = CUBLAS.scal!(length(A), x, A, 1)
bsub!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, B::CudaMatrix) = CUBLAS.axpy!(length(A), -1.0f0, B, 1, A, 1)
bsub!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, x::Float32) = (ccall((:add1,libkunet),Void,(Cint,Cfloat,Cmat),length(A),-x,A);A)

# # I could not get this to work:
# import Base: convert, promote_rule
# convert(::Type{Mat},x::Transpose{Mat})=x.obj
# promote_rule(::Type{Mat},::Type{Transpose{Mat}})=Mat

import Base: sum!, zeros, rand!, fill!  
# TODO: add error checking here since this is not a full implementation of sum!
sum!(r::CudaVecOrMat, A::CudaMatrix) = ccall((:bsum,libkunet),Void,(Cint,Cint,Cmat,Cmat),size(A,1),size(A,2),A,r) # reducedim.jl:226
zeros(A::CudaArray)=CUBLAS.scal!(length(A), zero(eltype(A)), copy(A), 1)
rand!(A::CudaArray)=(ccall((:randfill,libkunet),Void,(Cint,Cmat),length(A),A); A)
fill!(A::CudaArray,x::Float32)=(ccall((:fill,libkunet),Void,(Cint,Cfloat,Cmat),length(A),x,A); A)
gpuseed(n::Integer)=ccall((:gpuseed,libkunet),Void,(Culonglong,),convert(Culonglong, n))

# For debugging
function gpumem()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart.so"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    convert(Int,mfree[1])
end

end	########## CUDA extensions

