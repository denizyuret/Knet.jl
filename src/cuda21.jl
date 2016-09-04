cuda21 = [
("add","sum","ai+xi","xi","0"),
("mul","prod","ai*xi","xi","1"),
("max","maximum","(ai>xi?ai:xi)","xi","(-INFINITY)"),
("min","minimum","(ai<xi?ai:xi)","xi","INFINITY"),
]

function cuda21def(f, j=f, o...)
    J=Symbol(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$(S)_21"
        @eval begin
            function $J(x::KnetArray{$T}, region)
                if length(region) == ndims(x)
                    return fill!(similar(x,ntuple(i->1,ndims(x))), $J(x))
                elseif length(region) == ndims(x)-1
                    i0 = 0
                    ysize = ntuple(ndims(x)) do i
                        if in(i,region)
                            1
                        elseif i0>0
                            error("Bad region $region")
                        else
                            i0=i
                            size(x,i)
                        end
                    end
                    y = similar(x, ysize)
                    nx = length(x); ny = length(y); sy = stride(x,i0)
                    ccall(($F,$libknet8),Void,(Cint,Ptr{$T},Cint,Cint,Ptr{$T}),nx,x,sy,ny,y)
                    return y
                else
                    error("Only scalar and vector reductions supported.")
                end
            end
        end
    end
end

if isdefined(:libknet8)
    for f in cuda21
        isa(f,Tuple) || (f=(f,))
        cuda21def(f...)
    end
end
