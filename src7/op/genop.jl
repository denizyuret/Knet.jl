# Generator ops have no input.  There are four types:
# Par: writes output only at initialization.  Later modified using gradients by update!.
# Arr: writes output only at initialization.  Never changed, used for constant arrays.
# Rnd: generates random output every call.
# Input: copies the next network input to its output every call.

# Fields:
# dims: gives information about the output size, used by infersize
# init: an array or a random generator, used to fill the output
# initialized: used by Arr and Par to make sure output written only at initialization

abstract GenOp <: Op
type Par <: GenOp; dims; init; initialized; Par(; dims=nothing, init=nothing, o...)=new(dims,init,false); end
type Arr <: GenOp; dims; init; initialized; Arr(; dims=nothing, init=nothing, o...)=new(dims,init,false); end
type Rnd <: GenOp; dims; init; Rnd(; dims=nothing, init=nothing, o...)=new(dims,init); end
type Input <: GenOp; Input(;o...)=new(); end

forw(p::Par,y;o...)=(p.initialized ? y : (p.initialized=true; genforw(p.init,y)))
forw(p::Arr,y;o...)=(p.initialized ? y : (p.initialized=true; genforw(p.init,y)))
forw(p::Rnd,y;o...)=genforw(p.init,y)
forw(::Input,y;o...)=error("Input has no forward function, forw(::Net) handles it.")
back(p::GenOp,y;o...)=error("Back should not be called on inputless $(typeof(p))")

genforw(init::BaseArray,y)=copysync!(y,init)
genforw(init::Rgen,y)=rgen!(init,y)
genforw(init,y)=error("Unknown init type: $init")

ninputs(::GenOp)=0
canoverwrite(::GenOp)=false
back_reads_x(::GenOp)=false
back_reads_y(::GenOp)=false

# We have three sources of information to infer the output size.
# ysize is the size of the output register.
# p.dims may be specified by the user.
# the size of p.init if it is an array (could be a random distribution)
# We need to check their consistency and possibly fill in missing information.
function infersize(p::GenOp,ysize)
    dims = inferhelp(nothing, ysize)
    isdefined(p,:dims) && (dims = inferhelp(dims, p.dims))
    isdefined(p,:init) && isa(p.init,BaseArray) && (dims = inferhelp(dims, size(p.init)))
    return (dims == nothing ? nothing : tuple(dims))
end

function inferhelp(a::Dims,b::Dims)
    length(a)==length(b) || throw(DimensionMismatch())
    map(a,b) do ai,bi
        ai == bi ? ai :
        ai == 0  ? bi :
        bi == 0  ? ai :
        throw(DimensionMismatch())
    end
end

inferhelp(::Void,::Void)=nothing
inferhelp(a::Dims,::Void)=a
inferhelp(::Void,b::Dims)=b


