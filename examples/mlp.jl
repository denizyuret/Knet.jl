function mlp(; layers=(),
             loss=softmax, 
             actf=relu, 
             winit=Gaussian(0,.01), 
             binit=Constant(0))
    prog = quote
        x0 = input()
    end
    N = length(layers)
    for n=1:N
        x1 = symbol("x$(n-1)")
        x2 = symbol("x$n")
        op = :($x2 = layer($x1; n=$(layers[n]), f=$(n<N ? actf : loss), winit=$winit, binit=$binit))
        push!(prog.args, op)
    end
    return prog
end

layer(; n=1,
      f=relu,
      winit=Gaussian(0,.01),
      binit=Constant(0)) = 
quote
    x1 = input()
    w1 = par($n,0; init=$winit)
    x2 = dot(w1,x1)
    b2 = par(0; init=$binit)
    x3 = add(b2,x2)
    y3 = $(symbol(f))(x3)
end

softmax() = quote
    x1 = input()
    x2 = soft(x1)
    x3 = softloss(x2)
end

