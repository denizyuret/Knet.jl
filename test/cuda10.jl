using Base.Test, Knet

rand10(f,t,d...)=rand(t,d...)*t(0.8)+t(0.1)

cuda10flist = Any[+,-,*,/,\,.^]
for f in Knet.cuda10
    if isa(f,Tuple); f=f[2]; end
    if f=="stoa" || f=="atos"; continue; end
    push!(cuda10flist, eval(parse(f)))
end

no_sa = [ /, ]
no_as = [ \, ]

@testset "cuda10" begin
    for f in cuda10flist
        f1(x) = f(x[1],x[2])
        for t in (Float32, Float64)
            sx = rand10(f,t)+t(1)
            for n in (1,(1,1),2,(2,1),(1,2),(2,2))
                # @show f,t,n
                ax = rand10(f,t,n)
                !in(f,no_as) && (@test gradcheck(f1, [ax, sx]))
                !in(f,no_sa) && (@test gradcheck(f1, [sx, ax]))
                if gpu() >= 0
                    gx = KnetArray(ax)
                    !in(f,no_as) && (@test isapprox(Array{t}(f(ax,sx)),Array{t}(f(gx,sx))))
                    !in(f,no_sa) && (@test isapprox(Array{t}(f(sx,ax)),Array{t}(f(sx,gx))))
                    !in(f,no_sa) && (@test gradcheck(f1, [sx, gx]))
                    !in(f,no_as) && (@test gradcheck(f1, [gx, sx]))
                end
            end
        end
    end
end

nothing




# using Knet
# libknet8handle = Libdl.dlopen(Knet.libknet8)

# SIZE = 100000
# ITER = 100000
# x32 = KnetArray(rand(Float32,SIZE))
# y32 = similar(x32)
# s32 = rand(Float32)
# x64 = KnetArray(rand(Float64,SIZE))
# y64 = similar(x64)
# s64 = rand(Float64)

# function cuda10test(fname, jname=fname, o...)
#     println(fname)
#     fcpu = eval(parse(jname))
#     f32 = Libdl.dlsym(libknet8handle, fname*"_32_10")
#     @time cuda10rep(f32,x32,s32,y32)
#     isapprox(Array(y32),fcpu(Array(x32),s32)) || warn("$fname 32")
#     f64 = Libdl.dlsym(libknet8handle, fname*"_64_10")
#     @time cuda10rep(f64,x64,s64,y64)
#     isapprox(Array(y64),fcpu(Array(x64),s64)) || warn("$fname 64")
# end

# function cuda10rep{T}(f,x::KnetArray{T},s::T,y::KnetArray{T})
#     n = Cint(length(y))
#     for i=1:ITER
#         ccall(f,Void,(Cint,Ptr{T},T,Ptr{T}),n,x,s,y)
#     end
#     Knet.@cuda(cudart,cudaDeviceSynchronize,())
#     Knet.@cuda(cudart,cudaGetLastError,())
# end

# for f in Knet.cuda10
#     isa(f,Tuple) || (f=(f,))
#     cuda10test(f...)
#     cuda10test(f...)
#     cuda10test(f...)
# end
