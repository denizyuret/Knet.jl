# poly kernel: (s'*x+c)^d

type Poly <: Kernel; n; s; w; v; x; y; dy; c; d; 
# TODO: remove PercLoss, integrate into kernel
# TODO: use daume trick for kernel perceptron
# TODO: define all in kperceptron (mirroring perceptron.jl) and have kernel fn as a field
    function Poly(nclass;o...)
        l = setparam!(new(nclass); o...)
        isdefined(l,:c) || (l.c = 0)
        isdefined(l,:d) || (l.d = 1)
        return l
    end
end

# copy(l::Poly; o...)=Poly(s=copy(l.s), w=copy(l.w), v=copy(l.v), c=l.c, d=l.d)
setparam!(l::Poly; o...)=(for (n,v) in o; l.(n)=v; end; l)
kernel(l::Poly)=((l.s' * l.x + l.c) .^ l.d)
