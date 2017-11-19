"""
    dropout(x, p)

Given an array `x` and probability `0<=p<=1`, just return `x` if
testing, return an array `y` in which each element is 0 with
probability `p` or `x[i]/(1-p)` with probability `1-p` if training.
Training mode is detected automatically based on the type of `x`,
which is `AutoGrad.Rec` during gradient calculation.  Use the keyword
argument `training::Bool` to change the default mode and
`seed::Number` to set the random number seed for reproducible
results. See [(Srivastava et al. 2014)](http://jmlr.org/papers/v15/srivastava14a.html) 
for reference.

"""
function dropout(x,p; seed=0, training=false)
    if !training
        x
    elseif 0 < p < 1
        if seed != 0; setseed(seed); end
        dropout!(p,x,similar(x))
    elseif p == 0
        x
    elseif p == 1
        zeros(x)
    else
        error("Dropout probability not in [0:1]: $p")
    end
end

function dropback(dy,y,x,p)
    if 0 < p < 1
        dropback!(p,x,y,dy,similar(x))
    elseif p == 0
        dy
    elseif p == 1
        zeros(x)
    else
        error("Dropout probability not in [0:1]: $p")
    end
end

# Turn dropout into an AutoGrad primitive
dropout_r = recorder(dropout)
dropout(x::Rec,p; seed=0, o...)=dropout_r(x,p;seed=seed,training=true)
dropout(::Type{Grad{1}},d...;o...)=dropback(getval.(d)...) # d=dy,y,x,p

for S in (32,64)
    T = Symbol("Float$S")
    forw = Symbol("dropout_$S")
    back = Symbol("dropback_$S")
    @eval begin
        function dropout!(p::Number, x::KnetArray{$T}, y::KnetArray{$T})
            rand!(y)
            @knet8($forw,(Cint,$T,Ptr{$T},Ptr{$T}),length(y),$T(p),x,y)
            return y
        end
        function dropback!(p::Number, x::KnetArray{$T}, y::KnetArray{$T}, dy::KnetArray{$T}, dx::KnetArray{$T})
            @knet8($back,(Cint,$T,Ptr{$T},Ptr{$T},Ptr{$T},Ptr{$T}),length(dx),$T(p),x,y,dy,dx)
            return dx
        end
    end
end

function dropout!(p,x,y)
    rand!(y)
    p = convert(eltype(y),p)
    q = 1-p
    @inbounds for i=1:length(y)
        if y[i] > p
            y[i] = x[i] / q
        else
            y[i] = 0
        end
    end
    return y
end

function dropback!(p,x,y,dy,dx)
    p = convert(eltype(y),p)
    q = 1-p
    @inbounds for i=1:length(dx)
        if y[i] == 0
            dx[i] = 0
        else
            dx[i] = dy[i] / q
        end
    end
    return dx
end

