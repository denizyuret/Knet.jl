import Base: sum, prod, maximum, minimum, sumabs, sumabs2

cuda21 = [
("add","sum","ai+xi","xi","0"),
("mul","prod","ai*xi","xi","1"),
("max","maximum","(ai>xi?ai:xi)","xi","(-INFINITY)"),
("min","minimum","(ai<xi?ai:xi)","xi","INFINITY"),
("sumabs","sumabs","ai+xi","(xi<0?-xi:xi)","0"),
("sumabs2","sumabs2","ai+xi","(xi*xi)","0"),
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
                    error("Only scalar and vector reductions supported: $((size(x),region))")
                end
            end
        end
    end
end

for f in cuda21
    isa(f,Tuple) || (f=(f,))
    cuda21def(f...)
end


# The xentloss interface is no good because of double normalization.

"""
xentloss(x, p [,dims])

Compute cross-entropy loss for unnormalized log probability estimates
x and normalized probabilities p normalizing over the given
dimensions.  By default normalization is over the whole array, dims=1
normalizes over the columns, dims=2 normalizes over the rows of a 2-D
array.

"""
function xentloss(x,p,d...)
    x = x .- maximum(x,d...)
    z = log(sum(exp(x),d...))
    return sum(p .* (x .- z)) / length(z)
end

function xentback(x,p,d...)
    x = x .- maximum(x,d...)
    x = exp(x)
    z = sum(x,d...)
    return x./z - p
end

@primitive xentloss(x,p,d...),dy,y  (dy.*xentback(x,p,d...))

