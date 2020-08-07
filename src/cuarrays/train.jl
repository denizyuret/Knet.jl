# TODO: these are probably unnecessary, they are defined for AbstractArray
import ..Train20
using Knet: atype
Train20.update!(w::CuArray{T,N}, g, p) where {T,N} = Train20.gclip_update!(w, g, p)
Train20.param(x::CuArray; atype=atype()) = Param(atype(x))
Train20._optimizers(::CuArray{<:Number},otype; o...) = otype(;o...)
