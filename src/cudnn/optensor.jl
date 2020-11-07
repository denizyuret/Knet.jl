import CUDA.CUDNN: cudnnOpTensor, cudnnOpTensor!

cudnnOpTensor(x1::R,x2::R,d...;o...) where {R<:KnetArray} = cudnnOpTensor!(similar(x1, max.(size(x1),size(x2))),x1,x2,d...;o...)

cudnnOpTensor!(y::R,x1::R,x2::R,d...;o...) where {R<:KnetArray} = (cudnnOpTensor!(CuArray(y),CuArray(x1),CuArray(x2),d...;o...); y)

# TODO: define gradient
