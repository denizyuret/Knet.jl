import AutoGrad: ungetindex
using Knet.KnetArrays: DevArray, KnetArray
using AutoGrad: Value

function ungetindex(x::DevArray{T},dxi,i) where T
    if isbitstype(T)
        if dxi isa Value
            forw(addto!, zeroslike(x), forw(ungetindex, x, dxi, i))
        elseif recording()
            addtoindex!(zero(x), dxi, i...)
        else
            Sparse(x,Any[dxi],Any[i])
        end
    else
        # Using addtoindex! instead of setindex! to handle repeated indices
        addtoindex!(Array{Union{T,Nothing}}(nothing, size(x)), dxi, i...)
    end
end

