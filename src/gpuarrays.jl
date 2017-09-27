using GPUArrays, CLArrays
using Base.BLAS: axpy!
#CuArrays.allowslow(false)
const KnetArray = CLArray
const KnetVector{T} = KnetArray{T,1}
const KnetMatrix{T} = KnetArray{T,2}
export KnetArray, KnetVector, KnetMatrix

function logsumexp(x,d...)
    xmax = maximum(x,d...)
    xmax + log_dot(sum(exp_dot(x .- xmax),d...))
end

@primitive logsumexp(x,d...),dy,y  (dy .* exp_dot(x .- y))

function mat(x)
    if ndims(x) > 2
        xn = size(x,ndims(x))
        reshape(x, (div(length(x),xn),xn))
    elseif ndims(x)==2
        x
    elseif ndims(x)==1
        reshape(x, (length(x),1))
    else
        throw(MethodError(mat,x))
    end
end
