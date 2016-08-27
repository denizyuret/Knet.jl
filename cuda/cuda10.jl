using CUDArt
importall Base

cuda10 = [
# ("add",".+","s+x[i]"),
# ("sub",".-","s-x[i]"),
# ("mul",".*","s*x[i]"),
# ("div","./","s/x[i]"),
("pow",".^","pow(x[i],s)"),
# "hypot",
# "rhypot",
# "atan2",
# "frexp",
# "ldexp",
# "scalbn",
# "scalbln",
# "jn",
# "yn",
# "fmod",
# "remainder",
# "mod",
# "fdim",
]

function cuda10def(f, j=f, o...)
    libknet8 = Pkg.dir("Knet/cuda/libknet8")
    J=Symbol(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$(S)_10"
        @eval begin
            function $J(x::CudaArray{$T},s::$T)
                y = similar(x)
                ccall(($F,$libknet8),Void,(Cint,Ptr{$T},$T,Ptr{$T}),length(y),x,s,y)
                return y
            end
        end
    end
end
    
for f in cuda10
    isa(f,Tuple) || (f=(f,))
    cuda10def(f...)
end
