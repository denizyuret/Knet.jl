cuda11 = [
("add",".+","xi+yi"),
("sub",".-","xi-yi"),
("mul",".*","xi*yi"),
("div","./","xi/yi"),
("pow",".^","pow(xi,yi)"),
("max","max","(xi>yi?xi:yi)"),
("min","min","(xi<yi?xi:yi)"),
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

# familiar aliases for broadcasting operations of array & scalar (#7226):
(+){T}(x::KnetArray{T},y::KnetArray{T})=(.+)(x,y)
(-){T}(x::KnetArray{T},y::KnetArray{T})=(.-)(x,y)
#(*){T}(x::KnetArray{T},y::KnetArray{T})=(.*)(x,y) # This is matmul
#(/){T}(x::KnetArray{T},y::KnetArray{T})=(./)(x,y) # This is another linalg op


function cuda11def(f, j=f, o...)
    J=Symbol(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$(S)_11"
        @eval begin
            function $J(x::KnetArray{$T},y::KnetArray{$T})
                if size(x)==size(y)
                    z = tmplike(x)
                    ccall(($F,$libknet8),Void,(Cint,$Ptr{T},Ptr{$T},Ptr{$T}),length(z),x,y,z)
                    return z
                else
                    error("Not implemented yet.")
                end
            end
        end
    end
end

# Do this in cuda12, to handle size(x)!=size(y)
# for f in cuda11
#     isa(f,Tuple) || (f=(f,))
#     cuda11def(f...)
# end
