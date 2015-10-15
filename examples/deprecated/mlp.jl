function mlp(; layers=(), last=softmax, actf=relu, winit=Gaussian(0,.01), binit=Constant(0))
    prog = quote
        x0 = input()
    end
    N = length(layers)
    x0 = x1 = :x0
    for n=1:N
        x0 = x1; x1 = symbol("x$n")
        op = :($x1 = mlplayer($x0; n=$(layers[n]), f=$(n<N ? actf : last), winit=$winit, binit=$binit))
        push!(prog.args, op)
    end
    return prog
end

mlplayer(; n=1, f=relu, winit=Gaussian(0,.01), binit=Constant(0)) = 
quote
    x1 = input()
    w1 = par($n,0; init=$winit)
    x2 = dot(w1,x1)
    b2 = par(0; init=$binit)
    x3 = add(b2,x2)
    y3 = $f(x3)
end
