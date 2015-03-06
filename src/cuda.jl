# This file contains various compatibility fixes and hacks.  Hopefully
# it will shrink down to nothing as things get fixed in the original
# packages.

# arrays.jl:297, need this so generic code works with cpu arrays
import Base: copy!
copy!{T}(dst::DenseArray{T}, dstI::(Union(Int,Range1{Int})...), src::DenseArray{T}, srcI::(Union(Int,Range1{Int})...))=copy!(sub(dst, dstI...), sub(src, srcI...))

# when gc works these should not be necessary:
if isdefined(:CUDArt)
    import CUDArt: free, to_host
end
free(x)=x
to_host(x)=x


if isdefined(:CUDArt)   ########## CUDA extensions:

typealias Cmat Ptr{Float32}

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

# TODO: This does not seem to work:
gpuseed(n::Culonglong)=ccall((:gpuseed,libkunet),Void,(Culonglong,),n)

# For debugging
function gpumem()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart.so"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    convert(Int,mfree[1])
end

import Base.copy

function copy(l::Union(Layer,UpdateParam), to=nothing)
    ll = Layer()
    for n in names(l)
        isdefined(l,n) || continue
        iscnull(l.(n)) && continue
        if ((to == :cpu) && isa(l.(n), CudaArray))
            ll.(n) = to_host(l.(n))
        elseif ((to == :gpu) && isa(l.(n), AbstractArray))
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
iscnull(x)=(isa(x, CudaArray) && C_NULL==convert(typeof(C_NULL), x.ptr))

end	########## CUDA extensions

