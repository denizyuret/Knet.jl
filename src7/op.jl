# TODO: have a doc for ops here.
#"@knet function dot(w,x) is matrix multiplication."
#"@knet function input() fetches the next network input."
# ### mul2 element-wise multiplication:

abstract Op

# Each op must provide the following:
# back_reads_y (tosave)
# back_reads_x (tosave)
# ninputs (netcomp1)
# infersize (used by netinit)
# canoverwrite (used by initforw)
# ysize, loss (not used any more?)

forw(l::Op, y, x...; o...) = error("$(typeof(l)) has not implemented forw")
back(l::Op, dy, dx...; o...)   = error("$(typeof(l)) has not implemented back")
loss(l::Op, dy, y; o...)   = error("$(typeof(l)) has not implemented loss")

ninputs(l::Op)          = error("$(typeof(l)) has not implemented ninputs")
ysize(l::Op, x...)      = error("$(typeof(l)) has not implemented ysize")
canoverwrite(l::Op)       = error("$(typeof(l)) has not implemented canoverwrite")
back_reads_x(l::Op)     = error("$(typeof(l)) has not implemented back_reads_x")
back_reads_y(l::Op)     = error("$(typeof(l)) has not implemented back_reads_y")
# Base.eltype(::Op) = nothing

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

### DEAD CODE
# Each Op implements some common methods, stubs are given below.
# forw takes input x and returns output y, possibly setting some state.
# back takes dy, the loss gradient wrt y, calculates loss gradient wrt
# parameters and optionally returns dx, the loss gradient wrt x.
# Some layers overwrite their inputs.

