importall Base
using Knet,CUDArt
using AutoGrad
using AutoGrad: matmul2arg,broadcast2arg,unbroadcast,id2,math2arg,float2arg,broadcast2cmp
import AutoGrad: tofloat,_dbg

#include(Pkg.dir("Knet/src/util/gpu.jl"))
#include(Pkg.dir("Knet/src/tmplike.jl"))
#tmplike(a...)=similar(a...);tmpfree()=nothing;tmpmem()=nothing
# for f in ("cuda1","cuda01","cuda10","cuda11","cuda12","cuda22")
#     include(Pkg.dir("Knet/cuda/$f.jl"))
# end

tofloat{T<:AbstractFloat}(a::KnetArray{T})=a
_dbg{T}(x::KnetArray{T})=Symbol("K$(join([id2(x),size(x)...,8*sizeof(T)],'_'))")

for (f,d) in matmul2arg
    @eval @primitive $f(x1::KnetArray,x2::KnetArray)::y $(d[1]) $(d[2])
end
for (f,g) in broadcast2arg
    @eval @primitive $f(x1::KnetArray,x2::KnetArray)::y  unbroadcast(y,x1,$(g[1]))  unbroadcast(y,x2,$(g[2]))
    @eval @primitive $f(x1::Number,x2::KnetArray)::y  unbroadcast(y,x1,$(g[1]))  unbroadcast(y,x2,$(g[2]))
    @eval @primitive $f(x1::KnetArray,x2::Number)::y  unbroadcast(y,x1,$(g[1]))  unbroadcast(y,x2,$(g[2]))
end
for (f,g) in math2arg
    @eval @primitive $f(x1::KnetArray,x2::KnetArray)::y  unbroadcast(y,x1,dy->dy.*$(g[1]))  unbroadcast(y,x2,dy->dy.*$(g[2]))
    @eval @primitive $f(x1::Number,x2::KnetArray)::y  unbroadcast(y,x1,dy->dy.*$(g[1]))  unbroadcast(y,x2,dy->dy.*$(g[2]))
    @eval @primitive $f(x1::KnetArray,x2::Number)::y  unbroadcast(y,x1,dy->dy.*$(g[1]))  unbroadcast(y,x2,dy->dy.*$(g[2]))
end
for (f,g) in float2arg
    @eval @primitive $f(x1::KnetArray,x2::KnetArray)::y unbroadcast(y,x1,$(g[1])) unbroadcast(y,x2,$(g[2]))
    @eval @primitive $f(x1::Number,x2::KnetArray)::y unbroadcast(y,x1,$(g[1])) unbroadcast(y,x2,$(g[2]))
    @eval @primitive $f(x1::KnetArray,x2::Number)::y unbroadcast(y,x1,$(g[1])) unbroadcast(y,x2,$(g[2]))
end
for f in broadcast2cmp
    @eval begin
        @zerograd $f(x1::KnetArray,x2::KnetArray)
        @zerograd $f(x1::KnetArray,x2::Number)
        @zerograd $f(x1::Number,x2::KnetArray)
    end
end

@primitive  sum(x::KnetArray)  (dy->convert(eltype(x),dy).+zeros(x))
@primitive  sum(x::KnetArray,i...)  (dy->dy.+zeros(x))

include("profile.jl")
gpuinfo("before d0kn,w2kn")
d0kn = map(i->map(KnetArray,i),d0)
w2kn = map(KnetArray,w2)
gpuinfo("after d0kn,w2kn")
# w1kn = map(KnetArray,w1)
# timeall(w2kn,d0kn,10)
# loop(fun[9],w2kn,d0kn,10)
# (x1kn,y1kn)=first(d0kn)
# fun[10](w2kn,x1kn,y1kn)

function timeall_kn(w=w2kn,d=d0kn,t=10)
    for i=1:length(fun)
        print(i); printfun(fun[i])
        for j=1:3
            # gc_enable(false)
            # @time (loop_kn(fun[i],w,d,t); device_synchronize())
            @time loop_kn(fun[i],w,d,t)
            # gpuinfo("before gc")
            # gc_enable(true)
            # gc()
            sleep(2)
            # gpuinfo("after gc")
        end
    end
end

function loop_kn(f,w,d,t)
    for i in 1:t
        for (x,y) in d
            f(w,x,y)
        end
    end
end

# gc_enable(false); @time loop_kn(fun[9],w2kn,d0kn,10); gc_enable(true)
# gc_enable(false); @time loop_kn(fun[9],w2kn,d0kn,10); gc_enable(true)
# gc_enable(false); @time loop_kn(fun[9],w2kn,d0kn,10); gc_enable(true)
# gc_enable(false); @time loop_kn(fun[9],w2kn,d0kn,10); gc_enable(true)
# gc_enable(false); @time loop_kn(fun[9],w2kn,d0kn,10); gc_enable(true)
# gc_enable(false); @profile loop_kn(fun[9],w2kn,d0kn,10); gc_enable(true)
# Profile.print()
# Profile.print(format=:flat)

timeall_kn()
