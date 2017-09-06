using CuArrays
CuArrays.allowslow(false)
const KnetArray = CuArray
const KnetVector{T} = CuArray{T,1}
const KnetMatrix{T} = CuArray{T,2}
export KnetArray, KnetVector, KnetMatrix
