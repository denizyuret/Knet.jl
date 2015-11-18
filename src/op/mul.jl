type Mul <: Op; alpha; beta; end

mul(x1,x2,y; alpha=1, beta=1)=(Mul(alpha,beta),x1,x2,y)
ninputs(::Mul)=2
overwrites(::Mul)=false
back_reads_x(::Mul)=true
back_reads_y(::Mul)=false

function forw(l::Mul, x1, x2, y; o...)
    @assert x2 == nothing || size(x2) == size(y)
    if x1==nothing || x2==nothing # nothing represents zero
        nothing
    elseif size(x1) == size(x2)
        mul2!(l.alpha,x1,l.beta,x2,y)
    else
        ndims(x1) < ndims(x2) && (x1 = reshape_to_match(x1,x2))
        bmul!(l.alpha,x1,l.beta,x2,y)
    end
end

# z = mul(x,y; alpha,beta) = x^alpha * y^beta
# dJ/dx = dJ/dz dz/dx = dJ/dz * alpha * x^(alpha-1) * y^beta
# dJ/dy = dJ/dz dz/dy = dJ/dz * x^alpha * beta * y^(beta-1)

function back(l::Mul, dy, dx1, dx2; x=nothing, o...)
    if dx2 != nothing
        @assert size(dx2) == size(dy)
        if x[1] == nothing      # representing zero
            fill!(dx2, 0)
        elseif size(x[1]) == size(dy)
            mul2!(1,dy,l.alpha,x[1],dx2)
            l.beta == 1 || scale!(l.beta, mul2!(1,dx2,l.beta-1,x[2],dx2))
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
            mul2!(1,dy,l.beta,x[2],dx1)
            l.alpha == 1 || scale!(l.alpha, mul2!(1,dx1,l.alpha-1,x[1],dx1))
        else
            error("x2 and y should have the same size in mul")
        end
    else
        throw(DimensionMismatch())
    end
end

### mul2 element-wise multiplication:

mul2!(alpha::Number,a::Array,beta::Number,b::Array,c::Array)=(for i=1:length(c); c[i] = a[i]^alpha*b[i]^beta; end; c)
mul2!(alpha::Number,a::CudaArray{Float32},beta::Number,b::CudaArray{Float32},c::CudaArray{Float32})=(ccall((:mul2_32,libknet),Void,(Cint,Cfloat,Ptr{Cfloat},Cfloat,Ptr{Cfloat},Ptr{Cfloat}),length(a),Cfloat(alpha),a,Cfloat(beta),b,c); gpusync(); c)
mul2!(alpha::Number,a::CudaArray{Float64},beta::Number,b::CudaArray{Float64},c::CudaArray{Float64})=(ccall((:mul2_64,libknet),Void,(Cint,Cdouble,Ptr{Cdouble},Cdouble,Ptr{Cdouble},Ptr{Cdouble}),length(a),Cdouble(alpha),a,Cdouble(beta),b,c); gpusync(); c)

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

