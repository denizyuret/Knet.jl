import Base: dot

# TODO: Who implements averaging?  -- net should
# TODO: handle or don't use nothings? -- don't use

type Dot <: Op; end

"@knet function dot(w,x) is matrix multiplication."
dot(w,x,y)=(Dot(),w,x,y)
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

function infersize(d::Dot,a,b,c)
    # a,b may have more than 2 dims (convolution tensors etc.),
    # in which case we group the first n-1 into a super-column
    # a,b can also be nothing
    a==b==c==nothing && return nothing
    a==nothing || (a=[a...]; a1=prod(a[1:end-1]); a2=a[end])
    b==nothing || (b=[b...]; b1=prod(b[1:end-1]); b2=b[end])
    (c1,c2) = (c == nothing ? (0,0) :
               length(c)==2 ? c :
               throw(DimensionMismatch()))
    if a != nothing
        a1 == c1 ? nothing :
        a1 == 0  ? (length(a)==2 && (a[1]=c1)) :
        c1 == 0  ? c1=a1 :
        throw(DimensionMismatch())
    end
    if b != nothing
        b2 == c2 ? nothing :
        b2 == 0  ? b[end] = c2 :
        c2 == 0  ? c2 = b2 :
        throw(DimensionMismatch())
    end
    if a != nothing && b != nothing
        a2 == b1 ? nothing :
        a2 == 0  ? a[end]=b1 :
        b1 == 0  ? (length(b)==2 && (b[1]=a2)) :
        throw(DimensionMismatch())
    end
    (a==nothing ? a : tuple(a...),
     b==nothing ? b : tuple(b...),
     (c1,c2))
end
