# poly kernel: (s'*x+c)^d

type Poly <: Kernel; s; w; v; x; y; dy; c; d;
    function Poly(;o...)
        l = setparam!(new(); o...)
        isdefined(l,:w) || error("Poly: don't know how to initialize w")
        isdefined(l,:c) || (l.c = zero(eltype(x)))
        isdefined(l,:d) || (l.d = one(eltype(x)))
        return l
    end
end

copy(l::Poly; o...)=Poly(s=copy(l.s), w=copy(l.w), v=copy(l.v), c=l.c, d=l.d)
setparam!(l::Poly; o...)=(for (n,v) in o; l.(n)=v; end; l)
kernel(l::Poly)=((l.s' * l.x + l.c) .^ l.d)
