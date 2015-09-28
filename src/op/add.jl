# TODO: averaging? keep both arr and avg inside par, set output to one without copy?
# should be handled between net and par not by the op.
# DONE: handle nothings?  -- net is handling them.
# TODO: handle scalar input for adding a constant.
# TODO: back

type Add <: Op; end

add()=Add()
ninputs(::Add)=2
overwrites(::Add)=true
back_reads_x(::Add)=false
back_reads_y(::Add)=false

function forw(::Add, x1, x2, y; o...)
    @assert size(y) == size(x2)
    if size(x1) == size(x2)
        y===x2 ? axpy!(1,x1,y) :
        y===x1 ? axpy!(1,x2,y) :
        (copy!(y,x2); axpy!(1,x1,y))
    elseif length(x1) == 1
        y===x2 || copy!(y, x2)
        axpb!(1,x1[1],y)
    elseif size(x1) == biassize(y)
        biasforw(x1,x2,y)
    else
        error("Don't know how to add $(size(y))=$(size(x1))+$(size(x2))")
    end
end

biasforw(b::Vector, x::Array, y::Array)=(c=ndims(x)-1; for i=1:length(y); y[i] = x[i] + b[ind2sub(size(x),i)[c]]; end; y)
biasforw(b::Vector, x::Vector, y::Vector)=(for i=1:length(y); y[i] = x[i] + b[i]; end; y)
@gpu biasforw(b::CudaArray, x::CudaArray, y::CudaArray)=(y===x||copy!(y,x);cudnnAddTensor(b, y; mode=CUDNN_ADD_SAME_C))

function back(::Add, dy, dx1, dx2; o...)
    if dx2 != nothing
        @assert size(dx2) == size(dy)
        dx2 === dy || copy!(dx2, dy)
    end
    if dx1 == nothing
        # done
    elseif size(dx1) == size(dy)
        dx1 === dy || copy!(dx1, dy)
    elseif size(dx1) == biassize(dy)
        biasback(dy, dx1)
    elseif length(dx1) == 1
        error("not implemented") # TODO
    else
        error("Don't know how to add $(size(dy))=$(size(dx1))+$(size(dx2))")
    end
end

biasback(dy::Array, db::Vector)=(c=ndims(dy)-1; fill!(db, zero(eltype(db))); for i=1:length(dy); db[ind2sub(size(dy),i)[c]] += dy[i]; end)
biasback(dy::Vector, db::Vector)=(for i=1:length(dy); db[i]=dy[i]; end)
@gpu biasback(dy::CudaArray, db::CudaArray)=cudnnConvolutionBackwardBias(dy, db)

biassize(y)=(size(y, ndims(y)==1 ? 1 : ndims(y)-1),)

function infersize(::Add, x1, x2)
    if x1==x2==nothing
        nothing
    elseif x1==nothing
        (x1,x2,x2)
    elseif x2==nothing
        length(x1) > 1 ? (x1,x1,x1) : (x1,x2,x2)
    elseif length(x1) == 1
        if x1[1] == 1           # scalar addition
            (x1,x2,x2)
        elseif length(x2) == 1  # vector addition
            x3 = commonsize(x1,x2)
            (x3,x3,x3)
        else                    # bias addition
            x3 = sizeafterbias(x1,x2)
            ((x3[end-1],),x3,x3)
        end
    elseif length(x1) == length(x2) # element-wise
        x3 = commonsize(x1,x2)
        (x3,x3,x3)
    else
        error()
    end
end

function sizeafterbias(x1,x2)
    i1 = x1[1]
    i2 = x2[end-1]
    i1 == 0 && (i1=i2)
    i2 == 0 && (i2=i1)
    i1 == i2 || error()
    tuple(x2[1:end-2]..., i2, x2[end])
end

function commonsize(x1,x2)
    map(x1, x2) do i1,i2
        i1 == 0 && (i1=i2)
        i2 == 0 && (i2=i1)
        i1 == i2 || error()
        i1
    end
end
