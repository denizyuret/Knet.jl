abstract type Model end
# The following should be defined for a model:
# (f::Model)()
# (f::Model)(x)
# (f::Model)(x,y)
# (f::Model)(d::Data)

# Alternative functions:
# params(f::Model)    where {T<:Model} = try f(); catch e; error("params(::$T) should give an iterator over parameters."); end
# predict(f::Model,x) where {T<:Model} = try f(x); catch e; error("(::$T)(x) should be implemented as the predict function."); end
# loss(f::Model,x,y)  where {T<:Model} = try f(x,y); catch e; error("(::$T)(x,y) should be implemented as a loss function."); end
# loss(f::Model,d::Data) = mean(f(x[1],x[2]) for x in d)
