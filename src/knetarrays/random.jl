import Random: rand!, randn!
using Knet.KnetArrays: KnetArray
rand!(a::KnetArray)=(rand!(CuArray(a)); a)
randn!(a::KnetArray)=(randn!(CuArray(a)); a)
