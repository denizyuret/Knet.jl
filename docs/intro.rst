***********************
A Tutorial Introduction
***********************

We will begin by a quick tutorial on Knet.

Installation
------------

First download and install the latest version of Julia from
`<http://julialang.org/downloads>`_.  Type ``julia`` at the command
prompt to start the Julia interpreter.  To install Knet type
``Pkg.clone("git://github.com/denizyuret/Knet.jl.git")`` and go get
some coffee while Julia downloads and installs all the necessary
packages::

    $ julia
                   _
       _       _ _(_)_     |  A fresh approach to technical computing
      (_)     | (_) (_)    |  Documentation: http://docs.julialang.org
       _ _   _| |_  __ _   |  Type "?help" for help.
      | | | | | | |/ _` |  |
      | | |_| | | | (_| |  |  Version 0.4.2 (2015-12-06 21:47 UTC)
     _/ |\__'_|_|_|\__'_|  |  Official http://julialang.org/ release
    |__/                   |  x86_64-apple-darwin13.4.0
    
    julia> Pkg.clone("git://github.com/denizyuret/Knet.jl.git")

If you have a GPU machine, you may need to type ``Pkg.build("Knet")``
to compile the Knet GPU kernels.  If you do not have a GPU machine,
you don't need ``Pkg.build`` but you may get some warnings indicating
the lack of GPU support.  These can typically be safely ignored.  To
make sure everything has installed correctly, type
``Pkg.test("Knet")`` which should take a couple of minutes kicking the
tires.  If all is OK, continue with the next section, if not you can
get help at the `knet-users
<https://groups.google.com/forum/#!forum/knet-users>`_ mailing list.

Knet Functions
--------------
.. @kfun, compile, forw, get

To start using Knet, type ``using Knet`` at the Julia prompt.

.. doctest::

   julia> using Knet
   ...

In Knet, a machine learning model is defined using a function written
in a special syntax.  The following example defines a Knet function
for a simple linear regression model. You can type this definition at
the Julia prompt, or you can copy and paste it into a file which can
be loaded into Julia using ``include("filename")``:

.. testcode::

    @knet function lin(x)
        w = par(init=randn(3,5))
        b = par(init=randn(3,1))
        return w * x + b
    end

.. testoutput:: :hide:

   ...

In this definition:

- ``@knet`` indicates this is a special Knet function, not a regular
  Julia function.
- Everything between ``function`` and ``end`` gives the definition.
- ``lin`` is the function name and ``x`` is its only input argument.
- ``w`` and ``b`` are model parameters as indicated by the ``par``
  constructor.
- ``init`` is a keyword argument to ``par`` describing how the
  parameter should be initialized.
- ``randn`` is a Julia function that returns an array with given
  dimensions filled with random numbers from the standard normal
  distribution.
- The final ``return`` statement specifies the output of the Knet
  function, where ``*`` indicates matrix product and ``+`` indicates
  matrix addition.

It is important to note some important differences between a @knet
function like ``lin`` and regular Julia functions (which are defined
using a similar syntax but without the ``@knet`` macro).  First of all
only specific Knet operators can be used in @knet function definitions
(e.g. ``par``, ``*`` and ``+`` above, TODO: link to full list).
Secondly, ``lin`` does not get defined as an actual Julia function:

.. doctest::

   julia> lin(5)
   ERROR: UndefVarError: lin not defined

Instead, ``lin`` can be used to construct a machine learning model
that can be trained with examples and used for predictions.  In order
to turn ``lin`` into a model, we need to compile it:

.. doctest:: :hide:

    julia> srand(42);

.. doctest::

    julia> f = compile(:lin);
    
This defines ``f`` as an actual model (model or Net?) that we can
train and use for predictions (repeated).  Note that the colon
character preceding the name of our Knet function is required in the
compile expression.  (TODO: can we get rid of the colon with a macro?)
(TODO: The motivation behind this two step process, first defining a
Knet function then compiling it into a model, will become more clear
when we introduce compile time parameters.)

To test our model let's give it some input.  ``w`` is a 3x5 matrix, so
the input ``x`` should be 5 dimensional:

.. doctest::

    julia> x = randn(5,1)
    5x1 Array{Float64,2}:
      0.410653
     -0.85635
     -1.05099
      0.502079
     -0.216248

To obtain the prediction of ``f`` on the input ``x`` we use the
``forw`` function, which calculates ``w * x + b``:

.. doctest::     
    
    julia> forw(f,x)
    3x1 Array{Float64,2}:
      0.0261152
     -0.963565
      2.19213

We can query the model and see its parameters using ``get``:
      
.. doctest::     

    julia> get(f,:w)
    3x5 Array{Float64,2}:
     -0.556027   -0.299484  -0.468606  1.00331   0.518149
     -0.444383    1.77786    0.156143  1.08238   1.49138 
      0.0271553  -1.1449    -2.64199   0.187028  0.367563
    
We can also query the other parameter with ``get(f,:b)``, look at the
input with ``get(f,:x)``, reexamine the output using the special
``:return`` symbol with ``get(f,:return)``.  Note how we escape Knet
variable and function names using the colon character, e.g. ``:w``,
``:lin``.

So far it looks like a Knet model is a very complicated way to define
a very simple function.  In fact we can confirm that our model gives
us the same answer as the equivalent Julia expression:

.. doctest::     

    julia> get(f,:w) * get(f,:x) + get(f,:b)
    3x1 Array{Float64,2}:
      0.0261152
     -0.963565 
      2.19213  

TODO: motivate knet functions.      

.. - kfun as model: linear regression.
.. - kfun as new ops: mnist lenet.
.. - compile time parameters: 
.. - runtime parameters: conditionals: dropout? on mnist lenet?
.. - rbw registers: rnn intro, rnnlm (char based).
.. - conditionals: copyseq or adding or dropout?
.. 
.. - linear regression?  uci?  https://archive.ics.uci.edu/ml/datasets/Housing
.. - or do we do artificial data generation: cpu/gpu conversion may be difficult.
.. - mnist definitely
.. - mnist4d for convolution
.. - maybe something else for simple nnet?
.. - copyseq to introduce rnns
.. 
.. TODO:
.. 
.. - we need to talk about installation somewhere.
.. - Other requirements like v0.4.0, cuda libraries, cpu compatibility etc.
.. - DONE: Install latest v0.4.2.
.. - DONE: Update packages.
.. - DONE: Figure out no-gpu installation (CUDA* requirements)
.. - Create an amazon aws image for easy gpu work.
.. .. see http://sphinx-doc.org/ext/doctest.html
.. .. testcode for regular doctest for prompted examples
.. .. http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html#directives
