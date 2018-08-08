# Note that we tried and failed to automate the detection of "train"
# mode looking at the type of argument.  The argument is of type Rec
# only during training and only if it is a value influenced by model
# weights.  However people typically apply dropout to the input which
# is not a Rec.  So we are going back to no automation, make sure to
# supply p=0 during testing to stop dropout.

"""
    dropout(x, p)

Given an array `x` and probability `0<=p<=1`, just return `x` if
`p==0`, or return an array `y` in which each element is 0 with
probability `p` or `x[i]/(1-p)` with probability `1-p`.  Use
`seed::Number` to set the random number seed for reproducible
results. See [(Srivastava et al. 2014)](http://www.jmlr.org/papers/v15/srivastava14a.html)
for a reference.

"""
function dropout(x,p; seed=0)
    if 0 < p < 1
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
dropout(x::Rec,p;seed=0)=dropout_r(x,p;seed=seed)
dropout(::Type{Grad{1}},d...;o...)=dropback(getval.(d)...) # d=dy,y,x,p

# GPU implementation
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

# CPU implementation
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


"""
    alpha_dropout(x, p)

Dropout associated to the `selu` activation. 

Paper Ref.:
Self-Normalizing Neural Networks
https://arxiv.org/abs/1706.02515
"""
function alpha_dropout(x, p)
    training = x isa Rec
    (p == 0 || !training) && return x

    alpha = Float32(-1.758099)
    q = Float32(1-p)
    x = q*dropout(x .- alpha, p) .+ alpha #set dropped input to alpha
    a = 1 / sqrt(q + alpha^2 * q*p)
    b = -a * alpha * p
    return a*x + b
end

