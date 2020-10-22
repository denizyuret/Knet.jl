using Knet: atype
using Knet.KnetArrays: KnetArray
param(x::KnetArray; atype=atype()) = Param(convert(atype,x))
update!(w::KnetArray, g, p) = gclip_update!(w, g, p)
_optimizers(::KnetArray,otype; o...) = otype(;o...)
