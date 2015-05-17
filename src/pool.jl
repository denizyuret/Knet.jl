type Pool <: Layer; pd; x; y; dx; dy; Pool()=new(); end

Pool(pd)=error("CPU Pool not implemented.")
forw(l::Pool, x; o...)=error("CPU Pool not implemented")
back(l::Pool, dy; o...)=error("CPU Pool not implemented")
copy(l::Pool; o...)=Pool(l.pd)

# TODO: generalize to 3-D
# TODO: cpu implementation
# TODO: rethink the constructor interface

if GPU

Pool(d::Int,nd::Int=2)=(l=Pool();l.pd=PoolingDescriptor(fill(d,nd));l)
Pool(pd::PoolingDescriptor)=(l=Pool();l.pd=pd;l)

function forw(l::Pool, x::CudaArray; o...)
    initforw(l, x)
    cudnnPoolingForward(l.pd, l.x, l.y)
end

function initforw(l::Pool, x::CudaArray)
    l.x = x
    similar!(l, :y, l.x, cudnnGetPoolingNdForwardOutputDim(l.pd, l.x))
end

function back(l::Pool, dy::CudaArray; returndx=true, o...)
    @assert issimilar(dy, l.y)
    returndx || return
    initback(l, dy)
    cudnnPoolingBackward(l.pd, l.y, l.dy, l.x, l.dx)
end

function initback(l::Pool, dy::CudaArray)
    if (size(dy) == size(l.y))
        l.dy = dy
    else
        @assert length(dy) == length(l.y)
        l.dy = reshape(dy, size(l.y))
    end
    similar!(l, :dx, l.x)
end

end # if GPU
