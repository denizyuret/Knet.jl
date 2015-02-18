typealias Net Array{Layer,1}

function predict(net, x, batch=128)
    xrows,xcols = size(x)
    yrows,ycols = size(net[end].w, 1), xcols 
    y = similar(x, (yrows, ycols))
    xx = similar(net[1].w, (xrows, batch))
    for b = 1:batch:xcols
        e = b + batch - 1
        if e > xcols
            e = xcols
            xx = similar(net[1].w, (xrows, e-b+1))
        end
        copy!(xx, (1:xrows,1:e-b+1), x, (1:xrows,b:e))
        yy = xx
        for l = 1:length(net)
            yy = forw(net[l], yy)
        end
        copy!(y, (1:yrows,b:e), yy, (1:yrows,1:e-b+1))
    end
    y
end

function backprop(net, x, dy)
    for l = 1:length(net)
        x = forw(net[l], x)
    end
    for l = length(net):-1:1
        dy = back(net[l], dy, (l>1))
    end
end


# CUDA extensions:
import Base: copy!
copy!{T}(dst::DenseArray{T}, dstI::(Union(Int,Range1{Int})...), src::DenseArray{T}, srcI::(Union(Int,Range1{Int})...))=CUDArt.cudacopy!(dst, dstI, src, srcI)  # arrays.jl:297
