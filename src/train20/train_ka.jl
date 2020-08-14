import ..Train20
using ..Knet: atype
Train20.param(x::KnetArray; atype=atype()) = Param(atype(x))
Train20.update!(w::KnetArray{T,N}, g, p) where {T,N} = Train20.gclip_update!(w, g, p)
Train20._optimizers(::KnetArray{<:Number},otype; o...) = otype(;o...)
