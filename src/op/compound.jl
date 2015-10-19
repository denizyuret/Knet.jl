"""
@knet function wdot(x; out=0, winit=Gaussian(0,.01), o...) represents
a linear transformation (matrix product) w*x.  The output size can be
specified by the `out` parameter, and the weight matrix will be
initialized using the distribution or array given by winit.
"""
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

@knet function wf(x; f=relu, o...)
    y = wdot(x; o...)
    z = f(y; o...)
end

@knet function wbf(x; f=relu, o...)
    y = wdot(x; o...)
    z = bias(y; o...)
    a = f(z; o...)
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

"""
This is the LSTM model, originally proposed in "S. Hochreiter and
J. Schmidhuber. Long short-term memory. Neural Computation, 1997."

The keyword argument `out` determines the hidden size, `fbias`
determines the forget gate bias (0 by default), `winit` and `binit`
can be used to specify the default initialization for weight matrices
(Gaussian(0,.01) by default), and bias vectors (Constant(0) by
default).  Please see `@doc par` for details.

My implementation is closest to the one described in "Vinyals, O.,
Kaiser, L., Koo, T., Petrov, S., Sutskever, I., & Hinton,
G. (2014). Grammar as a foreign language. arXiv preprint
arXiv:1412.7449."  A popular variation is to include the so called
peephole connections, i.e. have the cell as an input to the forget,
output and input gates, see "A. Graves. Generating sequences with
recurrent neural networks. In Arxiv preprint arXiv:1308.0850, 2013."
This would replace the `add2(x,h)` operations with appropriately
defined `add3(x,h,cell)` operations below.

```
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
```
"""
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
recurrent weights are initialized with the identity matrix.  From: "Le,
Q. V., Jaitly, N., & Hinton, G. E. (2015). A Simple Way to Initialize
Recurrent Networks of Rectified Linear Units. arXiv preprint
arXiv:1504.00941."
```
@knet function irnn(x; scale=1, winit=Gaussian(0,.01), o...)
    wx = wdot(x; o..., winit=winit)
    wr = wdot(r; o..., winit=Identity(scale))
    xr = add(wx,wr)
    xrb = bias(xr; o...)
    r = relu(xrb)
end
```
"""
@knet function irnn(x; scale=1, winit=Gaussian(0,.01), o...)
    wx = wdot(x; o..., winit=winit)
    wr = wdot(r; o..., winit=Identity(scale))
    xr = add(wx,wr)
    xrb = bias(xr; o...)
    r = relu(xrb)
end

"""
@knet function repeat(x; frepeat=f, nrepeat=n, o...): gives the nth
repeated application of the @knet function f, which is defined to be
the function whose output at x is f(f(...(f(x))...)).  For example one
can define a multilayer perceptron with fixed sized layers using
repeat as follows:

```
@knet function wbf(x; f=relu, o...)
    y = wdot(x; o...)
    z = bias(y; o...)
    a = f(z; o...)
end

@knet function mlp(x; nlayers=0, actf=sigm, last=soft, o...)
    r = repeat(x; frepeat=wbf, nrepeat=nlayers-1, f=actf, o...)
    y = wbf(r; f=last, o...)
end
```
"""
function Base.repeat(x::Symbol, y::Symbol; frepeat=nothing, nrepeat=0, o...)
    @assert isa(frepeat,Function) && nrepeat > 0
    prog = Expr(:block)
    s0 = s1 = x
    for i=1:nrepeat-1
        s0 = s1
        s1 = gensym()
        push!(prog.args, :($s1 = $frepeat($s0; $o...)))
    end
    push!(prog.args, :($y = $frepeat($s1; $o...)))
    return prog
end
