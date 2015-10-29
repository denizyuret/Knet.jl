type Mul <: Op; Mul()=new(); end

# TODO: implement broadcasting
"@knet function mul(x,y) is element-wise multiplication."
mul(x1,x2,y)=(Mul(),x1,x2,y)
ninputs(::Mul)=2
overwrites(::Mul)=false
back_reads_x(::Mul)=true
back_reads_y(::Mul)=false

# x1,x2 is a pair of similarly sized input matrices

function forw(l::Mul, x1, x2, y; o...)
    @assert x2 == nothing || size(x2) == size(y)
    if x1==nothing || x2==nothing # nothing represents zero
        nothing
    elseif size(x1) == size(x2)
        mul2!(y,x1,x2)          # TODO: (minor) change order to x1,x2,y
    else
        error()
    end
end

function back(l::Mul, dy, dx1, dx2; x=nothing, o...)
    if dx2 != nothing
        @assert size(dx2) == size(dy)
        if x[1] == nothing      # representing zero
            fill!(dx2, 0)
        elseif size(x[1]) == size(dy)
            mul2!(dx2, dy, x[1])
        else
            throw(DimensionMismatch())
        end
    end
    if dx1 == nothing
        # done
    elseif size(dx1) == size(dy)
        if x[2] == nothing      # representing zero
            fill!(dx1, 0)
        elseif size(x[2]) == size(dy)
            mul2!(dx1, dy, x[2])
        else
            error("x2 and y should have the same size in mul")
        end
    else
        throw(DimensionMismatch())
    end
end

function infersize(m::Mul, x1, x2, y)
    if x1==x2==y==nothing
        nothing
    elseif x1==nothing || x2==nothing || y==nothing
        n = length(x1!=nothing ? x1 : x2!=nothing ? x2 : y!=nothing ? y : error())
        x1 == nothing && (x1 = ntuple(i->0,n))
        x2 == nothing && (x2 = ntuple(i->0,n))
        y == nothing && (y = ntuple(i->0,n))
        infersize(m,x1,x2,y)
    else
        length(x1)==length(x2)==length(y) || throw(DimensionMismatch())
        dims = map(x1,x2,y) do a,b,c
            n = 0
            a==0 || a==n ? nothing : n==0 ? n=a : throw(DimensionMismatch())
            b==0 || b==n ? nothing : n==0 ? n=b : throw(DimensionMismatch())
            c==0 || c==n ? nothing : n==0 ? n=c : throw(DimensionMismatch())
            n
        end
        (dims, dims, dims)
    end
end

