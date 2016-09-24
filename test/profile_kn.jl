importall Base
using Knet,AutoGrad
using Knet: gpuinfo, gpusync
gpu(true)

AutoGrad._dbg{T}(x::KnetArray{T})="K$(join([id2(x),size(x)...,8*sizeof(T)],'_'))"

include(Pkg.dir("AutoGrad/test/profile.jl"))
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
        for j=1:4
            gc_enable(false)
            @time (loop_kn(fun[i],w,d,t); gpusync())
            # @time loop_kn(fun[i],w,d,t)
            gpuinfo("before gc")
            gc_enable(true)
            gc()
            sleep(2)
            gpuinfo("after gc ")
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
