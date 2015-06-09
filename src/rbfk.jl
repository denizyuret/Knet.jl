# rbf kernel: exp(-gamma*|u-v|^2)

type Rbfk <: Kernel; s; w; v; x; y; dy; gamma;
    function Rbfk(;o...)
        l = setparam!(new(); o...)
        isdefined(l,:w) || error("Rbfk: don't know how to initialize w")
        isdefined(l,:gamma) || (l.gamma = one(eltype(x)))
        return l
    end
end

copy(l::Rbfk; o...)=Rbfk(s=copy(l.s), w=copy(l.w), v=copy(l.v), gamma=l.gamma)
setparam!(l::Rbfk; o...)=(for (n,v) in o; l.(n)=v; end; l)

function kernel(l::Rbfk)
    s2 = sum(l.s.^2, 1)
    x2 = sum(l.x.^2, 1)
    exp(-l.gamma * broadcast(+, x2, broadcast(+, s2', -2*(l.s' * l.x))))
end
