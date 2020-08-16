export param, param0, array_type
using AutoGrad: Param

"""
    param(array; atype)
    param(dims...; init, atype)
    param0(dims...; atype)

The first form returns `Param(atype(array))`.

The second form Returns a randomly initialized `Param(atype(init(dims...)))`.  

The third form `param0` is an alias for `param(dims...; init=zeros)`.

By default, `init` is `xavier_uniform` and `atype` is `Knet.atype()`.

"""
param,param0

# TODO: Knet.Param <: AutoGrad.Tracked as a separate type?
param(x::AbstractArray; atype=array_type[]) = Param(convert(atype,x))
param(d...; init=xavier_uniform, atype=atype())=Param(atype(init(d...)))
param0(d...; atype=atype())=param(d...; init=zeros, atype=atype)

"Default array type used by `param` and `param0`."
const array_type = Ref{Type}(Array{Float32})

