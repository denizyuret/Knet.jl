import Base: dot

# TODO: Who implements averaging?  -- net should
# TODO: handle or don't use nothings? -- don't use

type Dot <: Op; end

dot()=Dot()
ninputs(::Dot)=2
overwrites(::Dot)=false
back_reads_x(::Dot)=true
back_reads_y(::Dot)=false

function forw(::Dot, x1, x2, y; o...)
    if x1 == nothing || x2 == nothing
        return nothing
    end
    A_mul_B!(y, x1, x2)
end

function back(::Dot, dy, dx1, dx2; x=nothing, o...)
    dx1 != nothing && (x[2] != nothing ? A_mul_Bt!(dx1, dy, x[2]) : fill!(dx1, 0))
    dx2 != nothing && (x[1] != nothing ? At_mul_B!(dx2, x[1], dy) : fill!(dx2, 0))
end

function infersize(::Dot,a,b)
    ma = na = mb = nb = 0
    if a != nothing
        ma = prod(a[1:end-1])
        na = a[end]
    end
    if b != nothing
        mb = prod(b[1:end-1])
        nb = b[end]
    end
    na == 0 && (na = mb)
    mb == 0 && (mb = na)
    @assert na == mb
    ((ma,na), (mb,nb), (ma,nb))
end
