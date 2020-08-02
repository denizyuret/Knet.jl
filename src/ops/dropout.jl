export dropout
using Random: rand!
using Knet: training, seed!
using AutoGrad: AutoGrad, @primitive1, value

"""
    dropout(x, p; drop, seed)

Given an array `x` and probability `0<=p<=1` return an array `y` in which each element is 0
with probability `p` or `x[i]/(1-p)` with probability `1-p`. Just return `x` if `p==0`, or
`drop=false`. By default `drop=true` in a `@diff` context, `drop=false` otherwise.  Specify a
non-zero `seed::Number` to set the random number seed for reproducible results. See
[(Srivastava et al. 2014)](http://www.jmlr.org/papers/v15/srivastava14a.html) for a reference.

"""
function dropout(x,p; seed=0, drop=training())
    if !drop
        x
    elseif 0 < p < 1
        if seed != 0; seed!(seed); end
        dropout!(p,x,similar(x))
    elseif p == 0
        x
    elseif p == 1
        zero(x)
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
        zero(x)
    else
        error("Dropout probability not in [0:1]: $p")
    end
end

# Turn dropout into an AutoGrad primitive
@primitive1 dropout(x,p;seed=0,drop=training()),dy,y dropback(value.((dy,y,x,p))...)

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

# Note that we tried and failed to automate the detection of "train" mode looking at the type
# of argument.  The argument is of type Value only during training and only if it is a value
# influenced by model weights.  However people typically apply dropout to the input which is
# not a Value.  In 1.1.3 trying to automate again, this time using the @diff context: By
# default drop if AutoGrad is recording.

