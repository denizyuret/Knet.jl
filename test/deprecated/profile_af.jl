using ArrayFire
using AutoGrad
importall Base
using AutoGrad: matmul2arg,broadcast2arg,unbroadcast,id2,math2arg,float2arg,broadcast2cmp
import AutoGrad: tofloat,_dbg

tofloat{T<:AbstractFloat}(a::AFArray{T})=a
_dbg{T}(x::AFArray{T})=Symbol("AF$(join([id2(x),size(x)...,8*sizeof(T)],'_'))")
typealias AForN Union{AFArray,Number}
for (f,d) in matmul2arg
    @eval @primitive $f(x1::AFArray,x2::AFArray)::y $(d[1]) $(d[2])
end
for (f,g) in broadcast2arg
    @eval @primitive $f(x1::AFArray,x2::AFArray)::y  unbroadcast(y,x1,dy->dy.*$(g[1]))  unbroadcast(y,x2,dy->dy.*$(g[2]))
    @eval @primitive $f(x1::Number,x2::AFArray)::y  unbroadcast(y,x1,dy->dy.*$(g[1]))  unbroadcast(y,x2,dy->dy.*$(g[2]))
    @eval @primitive $f(x1::AFArray,x2::Number)::y  unbroadcast(y,x1,dy->dy.*$(g[1]))  unbroadcast(y,x2,dy->dy.*$(g[2]))
end
for (f,g) in math2arg
    @eval @primitive $f(x1::AFArray,x2::AFArray)::y  unbroadcast(y,x1,dy->dy.*$(g[1]))  unbroadcast(y,x2,dy->dy.*$(g[2]))
    @eval @primitive $f(x1::Number,x2::AFArray)::y  unbroadcast(y,x1,dy->dy.*$(g[1]))  unbroadcast(y,x2,dy->dy.*$(g[2]))
    @eval @primitive $f(x1::AFArray,x2::Number)::y  unbroadcast(y,x1,dy->dy.*$(g[1]))  unbroadcast(y,x2,dy->dy.*$(g[2]))
end
for (f,g) in float2arg
    @eval @primitive $f(x1::AFArray,x2::AFArray)::y unbroadcast(y,x1,$(g[1])) unbroadcast(y,x2,$(g[2]))
    @eval @primitive $f(x1::Number,x2::AFArray)::y unbroadcast(y,x1,$(g[1])) unbroadcast(y,x2,$(g[2]))
    @eval @primitive $f(x1::AFArray,x2::Number)::y unbroadcast(y,x1,$(g[1])) unbroadcast(y,x2,$(g[2]))
end
for f in broadcast2cmp
    @eval begin
        @zerograd $f(x1::AFArray,x2::AFArray)
        @zerograd $f(x1::AFArray,x2::Number)
        @zerograd $f(x1::Number,x2::AFArray)
    end
end

Base.eltype{T}(x::AFArray{T})=T
@primitive  sum(x::AFArray)  (dy->convert(eltype(x),dy).+zeros(x))
@primitive  sum(x::AFArray,i...)  (dy->dy.+zeros(x))
Base.size(x::AFArray,i)=(if i>ndims(x); 1; else; size(x)[i]; end)

include("profile.jl")
d0af = map(i->map(AFArray,i),d0)
# w1af = map(AFArray,w1)
w2af = map(AFArray,w2)
# timeall(w2af,d0af,10)
# loop(fun[9],w2af,d0af,10)
# (x1af,y1af)=first(d0af)
# fun[10](w2af,x1af,y1af)

function timeall_af(w=w2af,d=d0af,t=10)
    for i=1:length(fun)
        print(i); printfun(fun[i])
        for j=1:2
            sleep(2)
            gc_enable(false)
            @time (loop_af(fun[i],w,d,t); sync())
            gc_enable(true)
            @show deviceMemInfo()
        end
    end
end

function loop_af(f,w,d,t)
    for i in 1:t
        for (x,y) in d
            # ArrayFire.af_device_gc()
            f(w,x,y)
        end
    end
end

gc_enable(false); @time loop_af(fun[9],w2af,d0af,10); gc_enable(true)
gc_enable(false); @time loop_af(fun[9],w2af,d0af,10); gc_enable(true)
gc_enable(false); @time loop_af(fun[9],w2af,d0af,10); gc_enable(true)
gc_enable(false); @time loop_af(fun[9],w2af,d0af,10); gc_enable(true)
gc_enable(false); @time loop_af(fun[9],w2af,d0af,10); gc_enable(true)
gc_enable(false); @profile loop_af(fun[9],w2af,d0af,10); gc_enable(true)
Profile.print()
Profile.print(format=:flat)
