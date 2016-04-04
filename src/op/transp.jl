type Transp <: Op; Transp(;o...)=new();end

ninputs(::Transp)=1
canoverwrite(::Transp)=false
back_reads_x(::Transp)=false
back_reads_y(::Transp)=false

function transpforw(::Transp, x, y; o...)
  (y === x) && error("No array sharin in transp.")
  (x == nothing) && return nothing
  transpose!(y,x) # TODO decide dimensionality check, check cuda array compatibility
end


function transpback(::Transp, dx, dy; o...)
  if dx != nothing
    copysync!(dx, dy)
  end
end


function infersize(::Transp, xdims, ydims)
    if xdims==ydims==nothing
        nothing
    elseif xdims==nothing
        if length(ydims) == 2
          xdims=(ydims[2], ydims[1])
          return (xdims, ydims)
        else
          error("Only 2 dimensions can be handled")
        end
    elseif ydims==nothing
        if length(xdims) == 2
          ydims=(xdims[2], xdims[1])
          return (xdims, ydims)
        else
          error("Only 2 dimensions can be handled")
        end
    else
        @assert length(xdims) == length(ydims) == length(xdims) == 2
        if xdims[1] == ydims[2] && xdims[2] == ydims[1]
          return (xdims, ydims)
        else
          throw(DimensionMismatch())
        end
    end
end
