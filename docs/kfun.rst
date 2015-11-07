*******************************
The Anatomy of a @knet function
*******************************

Simple function::

    @knet function layer(x)
        w = par(dims=(100,0))
        b = par(dims=(0,))
        x1 = dot(w,x)
        x2 = add(b,x1)
        return relu(x2)
    end

* We start using the return statement instead of a variable.
* Make semicolon in par optional.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., &
Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural
networks from overfitting. The Journal of Machine Learning Research,
15(1), 1929-1958::

    @knet function drop(x)
        if training
            r = rnd(; rgen=Bernoulli(0.5))
            return mul(r,x)
        else
            return x
        end
    end

* The return statement can take just a variable?
* Without else, do we return x or do we return nothing (nothing is
  consistent with Julia).  We should stick with Julia semantics
  whenever possible.
* What if we return twice?  Early returns will need to terminate
  the forward pass or language restricted to single return.  If
  restricted to single return each branch can set the same variable
  that eventually is returned.  In either case ``x=y`` needs to be a
  legitimate instruction and only copy when necessary.

Le, Q. V., Jaitly, N., & Hinton, G. E. (2015). A Simple Way to
Initialize Recurrent Networks of Rectified Linear Units. arXiv
preprint arXiv:1504.00941 (IRNN,S2C)::

    @knet function irnn(x; hidden=0)
        wx = wdot(x; out=hidden)
        wr = wdot(r; out=hidden, winit=Identity(scale))
        xr = add(wx,wr)
        xrb = bias(xr; out=hidden)
        r = relu(xrb)
        if predicting
            return wb(r; out=1)
        end
    end

* The function may return nothing?  If predicting does not trigger.

Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence
learning with neural networks. Advances in neural information
processing systems, 3104-3112. (S2S)::

    @knet function s2s(word)
        if encoding
            wvec1 = wdot(word; out=hidden)
            hvec1 = lstm(wvec; out=hidden)
        else
            wvec2 = wdot(word; out=hidden)
            hvec2 = lstm(wvec; out=hidden)
            return wf(hvec2; out=vocab, f=soft)
        end
    end

* Problem: this won't work because the two lstms are not sharing hidden state.

S2S second attempt::

    @knet function lstm2(x,h)
        input  = add2(x,h; f=sigm)
        forget = add2(x,h; f=sigm)
        output = add2(x,h; f=sigm)
        newmem = add2(x,h; f=tanh)
        ig = mul(input,newmem)
        fc = mul(forget,cell)
        cell = add(ig,fc)
        tc = tanh(cell)
        return mul(tc,output)
    end        

    @knet function s2s(word)
        if encoding
            x = wdot(word; out=hidden)
            h = lstm2(x,h)
        else
            x = wdot(word; out=hidden)
            h = lstm2(x,h)
            return wf(h; out=vocab, f=soft)
        end
    end

* Make sure x=f(x) works.
* The two wdot and lstm2 keep their own weights.
* But they do share x and h (check this in compiler output).
* If we set x and h outside of the if statement, they'd also be
  sharing wdot and lstm weights.
* Unlike local variables x and h keep their state between calls, they
  are more like static variables in C.
* Need to figure out how to pass in the conditionals: we user regular
  parameters for runtime inputs, keyword argument for initialization.
  The compiler adds a final parameter for the output symbol.  The
  condition can be (1) a global variable, (2) the final parameter, (3)
  an optional parameter, (4) a keyword argument.
* We need to remember the conditions in the stack for back
  calculation.  So make conditions explicit inputs?  Do we handle this
  behind the scenes?
* Global condition seems to avoid complicating syntax, but
  semantically the condition is one of the inputs that determine the
  behavior of the funciton, so is hiding this going to cause trouble
  later?  Will we need other global inputs?  Will we need other runtime
  inputs that are not arrays?  Some alternatives::

    @knet function s2s(x, cond)
        if in(:training, cond)
	    ...
        if cond.training
	    ...
	if training
	    ...
	if cond[:training]
	    ...

* If we make cond an explicit parameter, will it also be passed down
  to child operations?
* How about if we pass a environment table of globals to make it more
  general?  We'd have undefined variable problem if we did not specify
  every condition.  A list of "true" symbols is more concise and serves the
  purpose right now.

Gutmann, M. U., & Hyvärinen, A. (2012). Noise-contrastive estimation
of unnormalized statistical models, with applications to natural image
statistics. The Journal of Machine Learning Research, 13(1),
307-361. (NCE)::

    @knet function nce(x, r; kqvec=nothing)
        h = lstm(x)
        w = par(dims=(vocab,0))
        b = par(dims=(vocab,1))
        if training
            q = arr(init=kqvec)
            rw = dot(r,w)
            rb = dot(r,b)
            rq = dot(r,q)
            y  = dot(rw,h)
            s  = add(rb,y)
            return nce(rq,s)
        else
            y = dot(w,h)
            s = add(b,y1)
            return soft(y2)
        end
    end

* We could define nce(x) and nce(x,r) as two functions but then cannot parameter share
* What do we pass for r when training, can we use r=nothing to make it optional?
* Insist on single return at the end?
* Should we pass q as an additional parameter?  No: Will result in copy
  every time.
* Compounds and operators would shorten the code significantly, e.g.
  ``return nce(r*q, r*b + (r*w)*h)``
* Use Julia operator names, i.e. ``.*`` for mul.
* Julia parses 2a into *(2,a).
* 2a+b is correctly parenthesized into +(*(2,a),b).
* 2*a*b is not parenthesized *(2,a,b) but 2a*b is turned into *(*(2,a),b).
* So handling compounds and arithmetic operators should be fairly simple.

Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification
with bidirectional LSTM and other neural network architectures. Neural
Networks, 18(5), 602-610. (BRNN)::

Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006,
June). Connectionist temporal classification: labelling unsegmented
sequence data with recurrent neural networks. In Proceedings of the
23rd international conference on Machine learning (pp. 369-376)
(CTC)::

Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective Approaches
to Attention-based Neural Machine Translation. arXiv preprint
arXiv:1508.04025. (Att)::

Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing
Machines. arXiv preprint arXiv:1410.5401. (NTM)::



** Draft **

Let us illustrate the basic components of a @knet function using the
following example::

    @knet function layer(x; out=0, f=relu, o...)
        w = par(; o..., dims=(out,0))
        b = par(; o..., dims=(0,))
        x1 = dot(w,x)
        x2 = add(b,x1)
        x3 = f(x2; o...)
    end

The definition starts with ``@knet function`` followed by the name of
the function.  Next comes the argument list which has several parts:

* Parameters before the semicolon denote the runtime inputs to the
  function.

* Keyword arguments after the semicolon are used to provide
  initialization parameters that customize the operators used in the
  function.

* A final parameter with three dots at the end denotes possible
  additional keyword arguments.

The important thing to remember is that *everything before the
semicolon is for the runtime*, and *everything after the semicolon is
for the compiler*.  The compiler uses the keyword arguments to
customize the operators in the function definition and they are never
used again.

The body of the function contains a sequence of Knet instructions.  It
is important to remember that *these instructions are not Julia
statements*.  They are very restricted, and are more like machine
language instructions than statements in a high level language.  Each
Knet instruction consists of a local variable, an equal sign, and an
operator with some arguments.

During the forward pass (?) the instructions are executed in the order
given, each instruction overwriting the value of the left-hand-side
variable.  The output of the function is the value of the last
variable set.  During the backward pass, each instruction computes the
loss gradient with respect to its inputs given the loss gradient with
respect to its output.

The operator of a Knet instruction can be a primitive (?), or another
user defined Knet function.  The argument syntax is similar to that of
a Knet function definition: runtime inputs before the semicolon, and
keyword arguments that specify initialization parameters after the
semicolon.  The values for the keyword arguments of an operator can
refer to constants or keyword arguments of the enclosing function but
not to any parameters or local variables.  Remember, parameters and
local variables change during runtime, keyword arguments are only used
during initialization.

.. Dropout

.. .. code::
.. @knet function drop(x; pdrop=0, o...)
..     if training
.. 	r = rnd(; rgen=Bernoulli(1-pdrop, 1/(1-pdrop)), testrgen=Constant(1))
.. 	y = mul(r,x)
..     end
.. end

.. Problem1: function empty if not training
.. Problem2: the return variable name is not fixed.

.. https://blog.twitter.com/2015/autograd-for-torch 
.. uses return statements
.. makes target variable explicit
.. f(params, input, target)
.. single input and target
.. params is a structure with weights and biases etc.

.. we should start with simpler examples and introduce keyword args,
.. o... etc later.

