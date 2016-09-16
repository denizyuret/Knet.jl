import Base: sum, prod, maximum, minimum, sumabs, sumabs2, countnz
import Base.LinAlg: norm, vecnorm

cuda20 = [
("add","sum","ai+xi","xi","0"),
("mul","prod","ai*xi","xi","1"),
("max","maximum","(ai>xi?ai:xi)","xi","(-INFINITY)"),
("min","minimum","(ai<xi?ai:xi)","xi","INFINITY"),
("sum1","sumabs","ai+xi","abs(xi)","0"),
("sum2","sumabs2","ai+xi","xi*xi","0"),
("nnz","countnz","ai+xi","(xi!=0)","0"),
]

norm(x::KnetVector, p::Real=2) = vecnorm(x, p)

function vecnorm{T}(x::KnetArray{T}, p::Real=2)
    if length(x) == 0
        zero(T)
    elseif p == 2
        sqrt(sumabs2(x))
    elseif p == 1
        sumabs(x)
    elseif p == Inf
        maximum(abs(x))
    elseif p == 0
        countnz(x)
    elseif p == -Inf
        minimum(abs(x))
    else
        sum(abs(x).^p)^(1/p)
    end
end

function cuda20def(f, j=f, o...)
    J=Symbol(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$(S)_20"
        @eval begin
            function $J(x::KnetArray{$T})
                ccall(($F,$libknet8),$T,(Cint,Ptr{$T}),length(x),x)
            end
        end
    end
end

#if isdefined(:libknet8)
    for f in cuda20
        isa(f,Tuple) || (f=(f,))
        cuda20def(f...)
    end
#end
