type Add2 <: Layer; Add2()=new(); end

ninputs(::Add2)=2

# x is a pair (Tuple{2}) of similarly sized input matrices
# if any of the input matrices is nothing, it represents the zero matrix, and we return the other matrix
# we'll overwrite the second matrix for the result

forw(l::Add2, x1, x2; o...)=(x1 == nothing ? x2 :
                             x2 == nothing ? x1 :
                             axpy!(1, x1, x2))

back(l::Add2, dy; o...)=(dy,dy)
