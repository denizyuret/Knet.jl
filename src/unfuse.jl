# The Broadcasted stuff below is a trick by @ylxdzsw to support julia v0.6.
# https://github.com/JuliaLang/julia/issues/22060#issuecomment-304294397

# This works in conjunction with broadcast methods in AutoGrad/src/unfuse.jl:

import Base: broadcast
using AutoGrad: Broadcasted # , broadcast_func

if VERSION >= v"0.6-"; @eval begin
    broadcast(f, x::Union{Number,AbstractArray,Rec,KnetArray}...)=f(Broadcasted.(x)...).value
end; end

function broadcast_func(f)
    if VERSION >= v"0.6-"
        bf = Symbol("broadcast#", lstrip(string(f), '.'))
        if isdefined(Knet, bf)
            # ok
        elseif isdefined(AutoGrad, bf)
            eval(Expr(:import, :AutoGrad, bf))
        else
            f = Symbol(f)
            if isdefined(Base, f)
                eval(Expr(:import, :Base, f))
            end
            @eval begin
                $bf(x...) = broadcast($f, x...)
                $f(x::Broadcasted...) = $bf(getval.(x)...) |> Broadcasted
            end
        end
    else
        bf = Symbol(f)
        if isdefined(Base, bf)
            eval(Expr(:import, :Base, bf))
        else
            warn("Base.$bf not defined")
        end
    end
    return bf
end


### DEAD CODE:

# Single function example:
# => sqrt.(x::KnetArray)   ## user
# => broadcast(sqrt, x)    ## base
# => sqrt(Broadcasted(x))  ## karray.jl:1070
# => broadcast#sqrt(x)     ## karray.jl:1084
# => sqrt(x)               ## unary.jl:84

# broadcast(f, x::KnetArray) = f(Broadcasted(x)).value
# broadcast(f, x1::KnetArray, x2::KnetArray) = f(Broadcasted(x1), Broadcasted(x2)).value
# broadcast(f, x1::KnetArray, x2) = f(Broadcasted(x1), x2).value
# broadcast(f, x1, x2::KnetArray) = f(x1, Broadcasted(x2)).value

# # Ambiguity fix
# broadcast(f, x1::KnetArray, x2::AutoGrad.Rec) = f(Broadcasted(x1), AutoGrad.Broadcasted(x2)).value
# broadcast(f, x1::AutoGrad.Rec, x2::KnetArray) = f(AutoGrad.Broadcasted(x1), Broadcasted(x2)).value

