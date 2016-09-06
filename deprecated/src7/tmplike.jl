# GPU memory allocation is very expensive.  So we create an
# application specific memory manager.  Typically same type, size, and
# number of arrays are needed in every iteration of training and
# testing.  During training these arrays should not be overwritten
# until the backward pass.  During testing they should be recycled as
# much as possible.

# We will have a Dict(length=>arrays) for each array type.

using CUDArt

type TmpStack; arr::Vector; idx::Int; TmpStack()=new([],0); end
typealias TmpDict Dict{Int,TmpStack}
for F in (Float16,Float32,Float64)
    A = CudaArray{F}
    let d=TmpDict()
        global tmpdict{T<:A}(::Type{T})=d
    end
end

tmpdict(a)=tmpdict(typeof(a))
tmpdict(T::Type)=error("TmpDict for $T not defined")

tmpfree()=(for T in (Float16,Float32,Float64), s in values(tmpdict(CudaArray{T})); s.idx=0; end)

function tmplike(a, dims::Dims=size(a))
    s = get!(TmpStack, tmpdict(a), prod(dims))
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

function tmpmem()
    a = []
    for T in (Float16,Float32,Float64)
        D = tmpdict(CudaArray{T})
        isempty(D) && continue
        push!(a, (T,:sizes,length(D),:arrays,sum(s->length(s.arr),values(D)),:nelem,sum(d->d[1]*length(d[2].arr),D)))
    end
    return a
end
