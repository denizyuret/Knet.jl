# Each Op implements some common methods, stubs are given below.
# forw takes input x and returns output y, possibly setting some state.
# back takes dy, the loss gradient wrt y, calculates loss gradient wrt
# parameters and optionally returns dx, the loss gradient wrt x.
# Some layers overwrite their inputs.

abstract Op <: Model

forw(l::Op, x...; o...) = error("$(typeof(l)) has not implemented forw")
back(l::Op, dy; o...)   = error("$(typeof(l)) has not implemented back")
loss(l::Op, dy; o...)   = error("$(typeof(l)) has not implemented loss")
params(l::Op)           = error("$(typeof(l)) has not implemented params")

ninputs(l::Op)          = error("$(typeof(l)) has not implemented ninputs")
ysize(l::Op, x...)      = error("$(typeof(l)) has not implemented ysize")
overwrites(l::Op)       = error("$(typeof(l)) has not implemented overwrites")
back_reads_x(l::Op)     = error("$(typeof(l)) has not implemented back_reads_x")
back_reads_y(l::Op)     = error("$(typeof(l)) has not implemented back_reads_y")

function Base.isequal(a::Op,b::Op)
    typeof(a)==typeof(b) || return false
    for n in fieldnames(a)
        if isdefined(a,n) && isdefined(b,n)
            isequal(a.(n), b.(n)) || return false
        elseif isdefined(a,n) || isdefined(b,n)
            return false
        end
    end
    return true
end
