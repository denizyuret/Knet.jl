@knet function wdot(x; out=0, winit=Gaussian(0,.01), o...)
    w = par(; o..., init=winit, dims=(out,0))
    y = dot(w,x)
end

@knet function bias(x; binit=Constant(0), o...)
    b = par(; o..., init=binit, dims=(0,))
    y = add(b,x)
end

@knet function wb(x; o...)
    y = wdot(x; o...)
    z = bias(y; o...)
end

@knet function wbf(x; f=relu, o...)
    y = wb(x; o...)
    z = f(y; o...)
end

@knet function wconv(x; out=0, window=0, cinit=Xavier(), o...)
    w = par(; o..., init=cinit, dims=(window, window, 0, out))
    y = conv(w,x)
end

@knet function cbfp(x; f=relu, cwindow=0, pwindow=0, o...)
    y = wconv(x; o..., window=cwindow)
    z = bias(y; o...)
    r = f(z; o...)
    p = pool(r; o..., window=pwindow)
end

@knet function add2(x1, x2; f=sigm, o...)
    y1 = wdot(x1; o...)
    y2 = wdot(x2; o...)
    x3 = add(y1,y2)
    y3 = bias(x3; o...)
    ou = f(y3; o...)
end

@knet function lstm(x; fbias=0, o...)
    input  = add2(x,h; o..., f=sigm)
    forget = add2(x,h; o..., f=sigm, binit=Constant(fbias))
    output = add2(x,h; o..., f=sigm)
    newmem = add2(x,h; o..., f=tanh)
    ig = mul(input,newmem)
    fc = mul(forget,cell)
    cell = add(ig,fc)
    tc = tanh(cell)
    h  = mul(tc,output)
end

"""
This is the IRNN model, a recurrent net with relu activations whose
recurrent weights are initialized with the identity matrix.  From: Le,
Q. V., Jaitly, N., & Hinton, G. E. (2015). A Simple Way to Initialize
Recurrent Networks of Rectified Linear Units. arXiv preprint
arXiv:1504.00941.
"""
@knet function irnn(x; scale=1, winit=Gaussian(0,.01), o...)
    wx = wdot(x; o..., winit=winit)
    wr = wdot(r; o..., winit=Identity(scale))
    xr = add(wx,wr)
    xrb = bias(xr; o...)
    r = relu(xrb)
end
