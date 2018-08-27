abstract type Model end
predict(f::T,x) where {T<:Model} = try f(x); catch e; error("(::$T)(x) should be implemented as the predict function."); end
loss(f::T,x,y)  where {T<:Model} = try f(x,y); catch e; error("(::$T)(x,y) should be implemented as a loss function."); end
params(f::T)    where {T<:Model} = error("params(::$T) should give an iterator over parameters.")
