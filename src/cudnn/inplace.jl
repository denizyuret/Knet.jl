# Defines set, scale, add for KnetArrays
# cudnnAddTensor only supports (a,b,c,d)+(1,1,c,1) and (a,b,c,d,e)+(1,1,1,d,1), use cudnnOpTensor instead.
# Compared to libknet8 x .+ b it is ~2x slower for (1,1,100,100), ~30% faster for (14,14,256,32)
# CUDA.jl x .+ b is 2x slower than both

import CUDA.CUDNN: cudnnSetTensor, cudnnScaleTensor, cudnnAddTensor

cudnnSetTensor(x::KnetArray, s; o...) = (cudnnSetTensor(CuArray(x), s; o...); x)

cudnnScaleTensor(x::KnetArray, s; o...) = (cudnnScaleTensor(CuArray(x), s; o...); x)

cudnnAddTensor(x::R, b::R; o...) where {R<:KnetArray} = (cudnnAddTensor(CuArray(x), CuArray(b); o...); x)

