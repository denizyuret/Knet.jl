export Op

struct Op; f; args; kwargs; end

Op(f,x...;o...) = Op(f, x, o)

(o::Op)(x) = o.f(x, o.args...; o.kwargs...)
