using Knet.KnetArrays: DevArray

using CUDA.CUDNN:
    cudnnSetTensor,
    cudnnScaleTensor,
    cudnnAddTensor,
    handle


function cudnnSetTensor!(x::R, s::Real;
                         xDesc::cudnnTensorDescriptor = TD(x)
                         ) where {T,R<:DevArray{T}}
    cudnnSetTensor(handle(), xDesc, x, Ref(T(s)))
    return x
end
                        

function cudnnScaleTensor!(x::R, s::Real;
                           xDesc::cudnnTensorDescriptor = TD(x)
                           ) where {T,R<:DevArray{T}}
    cudnnScaleTensor(handle(), xDesc, x, Ref(T(s)))
    return x
end


# Only supports (a,b,c,d)+(1,1,c,1) and (a,b,c,d,e)+(1,1,1,d,1), use cudnnOpTensor instead
# Compared to libknet8 x .+ b it is ~2x slower for (1,1,100,100), ~30% faster for (14,14,256,32)
# CUDA.jl x .+ b is 2x slower than both
function cudnnAddTensor!(x::R, b::R;
                         alpha::Real=1,
                         beta::Real=1,
                         bDesc::cudnnTensorDescriptor = TD(b),
                         xDesc::cudnnTensorDescriptor = TD(x),
                         ) where {T,N,R<:DevArray{T,N}}
    @assert N === 4 || N === 5
    @assert all(size(b,i) === (i === N-1 ? size(x,i) : 1) for i in 1:N)
    cudnnAddTensor(handle(), Ref(T(alpha)), bDesc, b, Ref(T(beta)), xDesc, x)
    return x
end
