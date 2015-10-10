wdot(; out=0, winit=Gaussian(0,.01), o...) = quote
    x = input()
    w = par($out,0; init=$winit, $o...)
    y = dot(w,x)
end

bias(; binit=Constant(0), o...) = quote
    x = input()
    b = par(0; init=$binit, $o...)
    y = add(b,x)
end

wbf(; out=0, f=relu, o...) = quote
    x = input()
    y = wdot(x; out=$out, $o...)
    z = bias(y; $o...)
    a = $f(z)
end

wconv(; out=0, window=0, cinit=Xavier(), o...) = quote
    x = input()
    w = par($window, $window, 0, $out; init=$cinit, $o...)
    y = conv(w,x)
end

convpool(; out=0, f=relu, cwindow=0, pwindow=0, o...) = quote
    x = input()
    y = wconv(x; out=$out, window=$cwindow, $o...)
    z = bias(y; $o...)
    r = $f(z)
    p = pool(r; window=$pwindow, $o...)
end

add2(; out=0, f=sigm, o...) = quote
    x1 = input()
    y1 = wdot(x1; out=$out, $o...)
    x2 = input()
    y2 = wdot(x2; out=$out, $o...)
    x3 = add(y1,y2)
    y3 = bias(x3; $o...)
    ou = $f(y3)
end

lstm(; out=0, fbias=0, o...) = quote
    x  = input()
    i  = add2(x,h;out=$out,f=sigm,$o...)
    f  = add2(x,h;out=$out,f=sigm,binit=$fbias,$o...)
    o  = add2(x,h;out=$out,f=sigm,$o...)
    g  = add2(x,h;out=$out,f=tanh,$o...)
    ig = mul(i,g)
    fc = mul(f,c)
    c  = add(ig,fc)
    tc = tanh(c)
    h  = mul(tc,o)
end

"""
This is the IRNN model, a recurrent net with relu activations whose
recurrent weights are initialized with the identity matrix.  From: Le,
Q. V., Jaitly, N., & Hinton, G. E. (2015). A Simple Way to Initialize
Recurrent Networks of Rectified Linear Units. arXiv preprint
arXiv:1504.00941.
"""
irnn(; out=0, o...) = quote
    x1 = input()
    x2 = wdot(x1; out=$out, $o...)
    x3 = wdot(re; out=$out, winit=Identity(), $o...)
    x4 = add(x2,x3)
    x5 = bias(x4; $o...)
    re = relu(x5)
end
