using CUDArt
importall Base

cuda12 = [
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

# Broadcast max/min haven't been defined:
max(a::Array,b::Array)=broadcast(max,a,b)
min(a::Array,b::Array)=broadcast(min,a,b)

function vbroadcast_shape(x,y)
    nz = max(ndims(x),ndims(y))
    dz = ones(Int,nz)
    xdims = ydims = xsame = ysame = xlast = ylast = 0; zlen = 1
    for i=1:nz
        if size(x,i) > 1
            xdims += 1; xlast = i
            dz[i] = size(x,i)
        end
        if size(y,i) > 1
            ydims += 1; ylast = i
            if dz[i] == 1
                dz[i] = size(y,i)
            else
                dz[i] == size(y,i) || throw(DimensionMismatch("arrays could not be broadcast to a common size"))
            end
        end
        xsame += (dz[i] == size(x,i))
        ysame += (dz[i] == size(y,i))
        zlen *= dz[i]
    end
    xsame == nz || xdims <= 1 || error("Only vector broadcasting supported")
    ysame == nz || ydims <= 1 || error("Only vector broadcasting supported")
    if xdims == 0
        sx = zlen; nx = 1
    elseif xdims == 1
        sx = prod(dz[1:xlast-1]); nx=dz[xlast]
    elseif xsame == nz
        sx = 1; nx=zlen
    else
        error("Broadcasting error")
    end
    if ydims == 0
        sy = zlen; ny = 1
    elseif ydims == 1
        sy = prod(dz[1:ylast-1]); ny=dz[ylast]
    elseif ysame == nz
        sy = 1; ny=zlen
    else
        error("Broadcasting error")
    end
    return (tuple(dz...), sx, nx, sy, ny)
end

function cuda12def(f, j=f, o...)
    libknet8 = Pkg.dir("Knet/cuda/libknet8")
    J=Symbol(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F11 = "$(f)_$(S)_11"
        F12 = "$(f)_$(S)_12"
        @eval begin
            function $J(x::KnetArray{$T},y::KnetArray{$T})
                if size(x)==size(y)
                    z = tmplike(x)
                    ccall(($F11,$libknet8),Void,(Cint,$Ptr{$T},Ptr{$T},Ptr{$T}),length(z),x,y,z)
                    return z
                else
                    (dz,sx,nx,sy,ny) = vbroadcast_shape(x,y)
                    z = tmplike(x,dz)
                    ccall(($F12,$libknet8),Void,(Cint,$Ptr{$T},Cint,Cint,Ptr{$T},Cint,Cint,Ptr{$T}),length(z),x,sx,nx,y,sy,ny,z)
                    return z
                end
            end
        end
    end
end
    
for f in cuda12
    isa(f,Tuple) || (f=(f,))
    cuda12def(f...)
end
