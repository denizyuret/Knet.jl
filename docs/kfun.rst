*******************************
The Anatomy of a @knet function
*******************************

Let us illustrate the basic components of a @knet function using the
following example:

.. code::
@knet function layer(x; out=0, f=relu, o...)
    w = par(; o..., dims=(out,0))
    b = par(; o..., dims=(0,))
    x1 = dot(w,x)
    x2 = add(b,x1)
    x3 = f(x2; o...)
end

The definition starts with `@knet function` followed by the name of
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

Simple function
---------------

.. code::
@knet function layer(x)
    w = par(; dims=(100,0))
    b = par(; dims=(0,))
    x1 = dot(w,x)
    x2 = add(b,x1)
    return relu(x2; o...)
end

Dropout
-------

.. code::
@knet function drop(x)
    if training
        r = rnd(; rgen=Bernoulli())
        return mul(r,x)
    else
        return x
    end
end

