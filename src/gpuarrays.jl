using GPUArrays
#CuArrays.allowslow(false)
const KnetArray = GPUArray
const KnetVector{T} = GPUArray{T,1}
const KnetMatrix{T} = GPUArray{T,2}
export KnetArray, KnetVector, KnetMatrix
