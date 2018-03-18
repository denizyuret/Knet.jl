"Module that contains the bindings of Knet functions."
module Kfun
using Knet: @knet, Add, Arr, Axpb, Conv, Copy, Dot, Input, LRN, Mul, NCE, Par, Pool, Relu, Rnd, Sigm, Soft, Soft73, Tanh

# function to add a knet function definition used by the @knet macro
kdef(x,y)=eval(Kfun,Expr(:(=),x,Expr(:quote,y)))

# Define some primitive knet functions as Op objects.
# The names on the LHS can be changed to modify the knet language without changing the implementation.
# Having these under the Kfun module (without import) allows use of Base names like dot or +.
* = Dot
+ = Add
.* = Mul
.+ = Add
add=Add
arr=Arr
axpb=Axpb
conv=Conv
copy=Copy
dot=Dot
input=Input
lrn=LRN
mul=Mul
nce=NCE
par=Par
pool=Pool
relu=Relu
rnd=Rnd
sigm=Sigm
soft=Soft
soft73=Soft73
tanh=Tanh

# Define some compound knet functions:

# """
# @knet function wdot(x; out=0, winit=Xavier(), o...) represents
# a linear transformation (matrix product) w*x.  The output size can be
# specified by the `out` parameter, and the weight matrix will be
# initialized using the distribution or array given by winit.
# """
@knet function wdot(x; out=0, winit=Xavier(), o...)
    w = par(; o..., init=winit, dims=(out,0))
    return w*x
end

# The interface changes marked with suffix 73 for now.
# Eventually these will become the defaults.
@knet function wdot73(x; out=0, o...)
    w = par(; o..., dims=(out,0))
    return w*x
end

# This is already defined as a primitive in actf.jl:
# @knet function copy(x; o...)
#     return axpb(x)              # TODO: do something more efficient.
# end

@knet function bias(x; binit=Constant(0), o...)
    b = par(; o..., init=binit, dims=(0,))
    return b+x
end

@knet function bias73(x; o...)
    b = par(; o..., dims=(0,))
    return b+x
end

@knet function wb(x; o...)
    y = wdot(x; o...)
    return bias(y; o...)
end

@knet function wb73(x; winit=Xavier(), binit=Constant(0), o...)
    y = wdot(x; o..., init=winit)
    return bias(y; o..., init=binit)
end

@knet function wf(x; f=:relu, o...)
    y = wdot(x; o...)
    return f(y; o...)
end

@knet function wbf(x; f=:relu, o...)
    h = wdot(x; o...)
    h = bias(h; o...)
    return f(h; o...)
end

@knet function wbf73(x; f=:relu, o...)
    h = wb(x; o...)
    return f(h; o...)
end

@knet function drop(x; pdrop=0, o...)
    if dropout
        return x .* rnd(init=Bernoulli(1-pdrop, 1/(1-pdrop)))
    else
        return x
    end
end

@knet function wconv(x; out=0, window=0, cinit=Xavier(), o...)
    w = par(; o..., init=cinit, dims=(window, window, 0, out))
    return conv(w,x; o...)
end

@knet function wconv73(x; out=0, window=0, o...)
    w = par(; o..., dims=(window, window, 0, out))
    return conv(w,x; o...)
end

@knet function bias4(x; binit=Constant(0), o...)
    b = par(; o..., init=binit, dims=(1,1,0,1))
    return b+x
end

@knet function bias473(x; o...)
    b = par(; o..., dims=(1,1,0,1))
    return b+x
end

@knet function cbfp(x; f=:relu, cwindow=0, pwindow=0, o...)
    y = wconv(x; o..., window=cwindow)
    z = bias4(y; o...)
    r = f(z; o...)
    return pool(r; o..., window=pwindow)
end

@knet function cbfp73(x; cwindow=3, cpadding=div(cwindow,2), cstride=1, cmode=Knet.CUDNN_CONVOLUTION,
                      pwindow=2, ppadding=0, pstride=pwindow, pmode=Knet.CUDNN_POOLING_MAX,
                      cinit=Xavier(), binit=Constant(0), f=:relu, o...)
    y = wconv(x; o..., window=cwindow, padding=cpadding, stride=cstride, mode=cmode, init=cinit)
    z = bias4(y; o..., init=binit)
    r = f(z; o...)
    return pool(r; o..., window=pwindow, padding=ppadding, stride=pstride, mode=pmode)
end

@knet function cbf73(x; cwindow=3, cpadding=div(cwindow,2), cstride=1, cmode=Knet.CUDNN_CONVOLUTION,
                     cinit=Xavier(), binit=Constant(0), f=:relu, o...)
    y = wconv(x; o..., window=cwindow, padding=cpadding, stride=cstride, mode=cmode, init=cinit)
    z = bias4(y; o..., init=binit)
    return f(z; o...)
end

# @knet function wbf2(x1, x2; f=:sigm, o...)
#     y = wdot(x2; o...) + wdot(x1; o...) # if (y1,y2) lstm cannot infer size with one column input
#     z = bias(y; o...)
#     return f(z; o...)
# end

# Go back to old def for debugging, should be equivalent:
@knet function wbf2(x1, x2; f=:sigm, o...)
    y1 = wdot(x1; o...)
    y2 = wdot(x2; o...)
    x3 = add(y2,y1)             # if (y1,y2) lstm cannot infer size with one column input
    y3 = bias(x3; o...)
    return f(y3; o...)
end

@knet function wbf273(x1, x2; f=:sigm, winit=Xavier(), binit=Constant(0), o...)
    y1 = wdot(x1; o..., init=winit)
    y2 = wdot(x2; o..., init=winit)
    x3 = add(y2,y1)             # if (y1,y2) lstm cannot infer size with one column input
    y3 = bias(x3; o..., init=binit)
    return f(y3; o...)
end

# """
# This is the LSTM model, originally proposed in "S. Hochreiter and
# J. Schmidhuber. Long short-term memory. Neural Computation, 1997."

# The keyword argument `out` determines the hidden size, `fbias`
# determines the forget gate bias (0 by default), `winit` and `binit`
# can be used to specify the default initialization for weight matrices
# (Xavier() by default), and bias vectors (Constant(0) by
# default).  Please see `@doc par` for details.

# My implementation is closest to the one described in "Vinyals, O.,
# Kaiser, L., Koo, T., Petrov, S., Sutskever, I., & Hinton,
# G. (2014). Grammar as a foreign language. arXiv preprint
# arXiv:1412.7449."  A popular variation is to include the so called
# peephole connections, i.e. have the cell as an input to the forget,
# output and input gates, see "A. Graves. Generating sequences with
# recurrent neural networks. In Arxiv preprint arXiv:1308.0850, 2013."
# This would replace the `wbf2(x,h)` operations with appropriately
# defined `add3(x,h,cell)` operations below.

# ```
# @knet function lstm(x; fbias=1, o...)
#     input  = wbf2(x,h; o..., f=:sigm)
#     forget = wbf2(x,h; o..., f=:sigm, binit=Constant(fbias))
#     output = wbf2(x,h; o..., f=:sigm)
#     newmem = wbf2(x,h; o..., f=:tanh)
#     ig = mul(input,newmem)
#     fc = mul(forget,cell)
#     cell = add(ig,fc)
#     tc = tanh(cell)
#     h  = mul(tc,output)
# end
# ```
# """
@knet function lstm(x; fbias=1, o...)
    input  = wbf2(x,h; o..., f=:sigm)
    forget = wbf2(x,h; o..., f=:sigm, binit=Constant(fbias))
    output = wbf2(x,h; o..., f=:sigm)
    newmem = wbf2(x,h; o..., f=:tanh)
    cell = input .* newmem + cell .* forget
    h  = tanh(cell) .* output
    return h
end

# Based on http://arxiv.org/pdf/1412.3555v1.pdf
# And https://github.com/karpathy/char-rnn/blob/master/model/GRU.lua
@knet function gru(x;o...)
    update = wbf2(x,h; o..., f=:sigm)
    reset  = wbf2(x,h; o..., f=:sigm)
    gatedHidden = mul(reset, h)
    hcandidate = wbf2(x,gatedHidden; o..., f=:tanh)
    zh = mul(hcandidate, update)
    minup = axpb( update;a=-1, b=1)
    zhm1 = mul(minup, h)
    h = add(zhm1, zh)
    return h
end
# This is the IRNN model, a recurrent net with relu activations whose
# recurrent weights are initialized with the identity matrix.  From: "Le,
# Q. V., Jaitly, N., & Hinton, G. E. (2015). A Simple Way to Initialize
# Recurrent Networks of Rectified Linear Units. arXiv preprint
# arXiv:1504.00941."
# ```
# @knet function irnn(x; scale=1, winit=Xavier(), o...)
#     wx = wdot(x; o..., winit=winit)
#     wr = wdot(r; o..., winit=Identity(scale))
#     xr = add(wx,wr)
#     xrb = bias(xr; o...)
#     r = relu(xrb)
# end
# ```

@knet function irnn(x; scale=1, winit=Xavier(), o...)
    wx = wdot(x; o..., winit=winit)
    wr = wdot(r; o..., winit=Identity(scale))
    r = relu(bias(wx + wr; o...))
    # xr = wx + wr
    # xrb = bias(xr; o...)
    # r = relu(xrb)
    return r
end

# """
# @knet function repeat(x; frepeat=f, nrepeat=n, o...): gives the nth
# repeated application of the @knet function f, which is defined to be
# the function whose output at x is f(f(...(f(x))...)).  For example one
# can define a multilayer perceptron with fixed sized layers using
# repeat as follows:

# ```
# @knet function wbf(x; f=:relu, o...)
#     y = wdot(x; o...)
#     z = bias(y; o...)
#     a = f(z; o...)
# end

# @knet function mlp(x; nlayers=0, actf=sigm, last=soft, o...)
#     r = repeat(x; frepeat=wbf, nrepeat=nlayers-1, f=actf, o...)
#     y = wbf(r; f=last, o...)
# end
# ```
# """

function repeat(; frepeat=nothing, nrepeat=0, o...)
    @assert isa(frepeat,Symbol) && nrepeat > 0
    x0 = x1 = gensym(:x)
    fname = gensym(:f)
    fhead = Expr(:call,fname,x0)
    fbody = Expr(:block)
    for i=1:nrepeat-1
        x0 = x1
        x1 = gensym(:x)
        push!(fbody.args, :($x1 = $frepeat($x0; $o...)))
    end
    push!(fbody.args, :(return $frepeat($x1; $o...)))
    return Expr(:function, fhead, fbody)
end

end # module
