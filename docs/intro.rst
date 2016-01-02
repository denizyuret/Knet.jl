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

Models, Functions, and Operators
--------------------------------
..
   @kfun, compile, forw, get

To start using Knet, type ``using Knet`` at the Julia prompt.

.. doctest::

   julia> using Knet
   ...

In Knet, a machine learning model is defined using a special function
syntax.  The following example defines a Knet function for a simple
linear regression model with 13 inputs and 1 output. You can type this
definition at the Julia prompt, or you can copy and paste it into a
file which can be loaded into Julia using ``include("filename")``:

.. testcode::

    @knet function lin(x)
        w = par(init=randn(1,13))
        b = par(init=randn(1,1))
        return w * x .+ b
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
  function, where ``*`` denotes matrix product and ``.+`` denotes
  elementwise addition.

This looks a lot like a regular Julia `function definition
<http://julia.readthedocs.org/en/release-0.4/manual/functions>`_
except for the ``@knet`` macro.  However it is important to emphasize
that the @knet macro does not define ``lin`` as a regular Julia
function or variable.  Furthermore, only a restricted set of statement
types (e.g. assignment and return statements) and operators
(e.g. ``par``, ``*`` and ``.+``) can be used in a @knet function
definition.  A full list of Knet primitive operators is given below:

===============================	==============================================================================
Operator                	Description
===============================	==============================================================================
:func:`par() <par>`		a parameter array, updated during training; kwargs: [#]_ ``dims, init``
:func:`rnd() <rnd>`		a random array, updated every call; kwargs: ``dims, init``
:func:`arr() <arr>`           	a constant array, never updated; kwargs: ``dims, init``
:func:`dot(A,B) <dot>`        	matrix product of ``A`` and ``B``; alternative notation: ``A*B``
:func:`add(A,B) <add>`		elementwise broadcasting [#]_ addition of arrays ``A`` and ``B``, alternative notation: ``A.+B``
:func:`mul(A,B) <mul>`        	elementwise broadcasting multiplication of arrays ``A`` and ``B``; alternative notation: ``A.*B``
:func:`conv(W,X) <conv>`       	convolution with filter ``W`` and input ``X``; kwargs: ``padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION``
:func:`pool(X) <pool>`		pooling; kwargs: ``window=2, padding=0, stride=window, mode=CUDNN_POOLING_MAX``
:func:`axpb(X) <axpb>`         	computes ``a*x^p+b``; kwargs: ``a=1, p=1, b=0``
:func:`copy(X) <copy>`         	copies ``X`` to output.
:func:`relu(X) <relu>`		rectified linear activation function: ``(x > 0 ? x : 0)``
:func:`sigm(X) <sigm>`		sigmoid activation function: ``1/(1+exp(-x))``
:func:`soft(X) <soft>`		softmax activation function: ``(exp xi) / (Σ exp xj)``
:func:`tanh(X) <tanh>`		hyperbolic tangent activation function.
===============================	==============================================================================

.. [#] Both Julia and Knet functions accept optional `keyword
       arguments
       <http://julia.readthedocs.org/en/release-0.4/manual/functions/#keyword-arguments>`_.
       Functions with keyword arguments are defined using a semicolon
       in the signature, e.g. ``plot(x, y; width=1, height=2)``, the
       semicolon is optional when the function is called,
       e.g. both ``plot(x, y, width=2)`` or ``plot(x, y; width=2)``
       work.  Unspecified keyword arguments take their default values
       specified in the function definition.

.. [#] `Broadcasting operations <http://julia.readthedocs.org/en/release-0.4/manual/arrays/#broadcasting>`_
       are element-by-element binary operations on arrays
       of possibly different sizes, such as adding a vector to each
       column of a matrix.  They expand singleton dimensions in array
       arguments to match the corresponding dimension in the other
       array without using extra memory, and apply the given function
       elementwise.

In order to turn ``lin`` into a machine learning model that can be
trained with examples and used for predictions, we need to compile it:

.. doctest:: :hide:

    julia> srand(42);

.. doctest::

    julia> f = compile(:lin);	# Note that the colon before lin is required
    ...
    
..
   This defines ``f`` as an actual model (model or Net?) that we can
   train and use for predictions (repeated).  Note that the colon
   character preceding the name of our Knet function is required in the
   compile expression.  (TODO: can we get rid of the colon with a macro?)
   (TODO: The motivation behind this two step process, first defining a
   Knet function then compiling it into a model, will become more clear
   when we introduce compile time parameters.)

To test our model let's give it some input.  ``w`` is a 1x13 row
vector, so the input ``x`` should be a 13x1 column vector:

.. doctest::

    julia> x = randn(13,1)
    13x1 Array{Float64,2}:
      0.367563
     -0.886205
     ...
      0.569829
     -1.42206

To obtain the prediction of model ``f`` on input ``x`` we use the
``forw`` function, which calculates ``w * x .+ b``:

.. doctest::     
    
    julia> forw(f,x)
    1x1 Array{Float64,2}:
     -1.00532

We can query the model and see its parameters using ``get`` (Note that
we need to escape Knet variable names using the `colon
character
<http://julia.readthedocs.org/en/release-0.4/manual/metaprogramming#symbols>`_,
just like we did for ``:lin`` when compiling.):
      
.. doctest::

    julia> get(f,:w)
    1x13 Array{Float64,2}:
     -0.556027  -0.444383  0.0271553 ... 1.08238  0.187028  0.518149

    julia> get(f,:b)
    1x1 Array{Float64,2}:
     1.49138
    
We can also look at the input with ``get(f,:x)``, reexamine the output
using the special ``:return`` symbol with ``get(f,:return)``.  In fact
using ``get``, we can confirm that our model gives us the same answer
as an equivalent Julia expression:

.. doctest::     

    julia> get(f,:w) * get(f,:x) .+ get(f,:b)
    1x1 Array{Float64,2}:
     -1.00532

..
   Also note that ``lin`` is not defined as a regular Julia function or
   variable.

   .. doctest::

      julia> lin(5)
      ERROR: UndefVarError: lin not defined

..
   So far it looks like all Knet gave us is a very complicated way to
   define a very simple function.  So why would anybody bother defining a
   @knet function with all the syntactic restrictions, limited number of
   operators, need for compilation etc.?

Training
--------

What makes a machine learning model different from an ordinary
function is its ability to learn from data.  Let us download the
`Housing <http://archive.ics.uci.edu/ml/datasets/Housing>`_ dataset
from the `UCI Machine Learning Repository
<http://archive.ics.uci.edu/ml/datasets.html>`_ to train our model:

.. doctest::
   
   julia> using Requests

   julia> url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data";

   julia> data = readdlm(get(url).data)'	# Don't forget the final apostrophe for transpose
   14x506 Array{Float64,2}:
   ...

The dataset has housing related information about 506 neighborhoods in
Boston, each with 14 attributes.  The last attribute is the median
house price to be predicted, so let's separate it:

.. doctest::
   
   julia> x = data[1:13,:]
   13x506 Array{Float64,2}:
   ...

   julia> y = data[14,:]
   1x506 Array{Float64,2}:
   ...

You may have noticed that the input attributes have very different
ranges.  It is usually a good idea to normalize them:

.. doctest::

   julia> x = (x .- mean(x,2)) ./ std(x,2)
   13x506 Array{Float64,2}:
   ...

It is also a good idea to split our dataset into train and test
portions so we can estimate how well our model will do on unseen data:

.. doctest::

   julia> r = randperm(size(x,2))
   506-element Array{Int64,1}:
   ...

   julia> xtrn=x[:,r[1:400]]
   13x400 Array{Float64,2}:
   ...
    
   julia> ytrn=y[:,r[1:400]]
   1x400 Array{Float64,2}:
   ...
    
   julia> xtst=x[:,r[401:end]]
   13x106 Array{Float64,2}:
   ...
    
   julia> ytst=y[:,r[401:end]]
   1x106 Array{Float64,2}:
   ...
    
Let's see how well our randomly initialized model does before
training:

.. doctest::

   julia> ypred = forw(f, xtst)
   1x106 Array{Float64,2}:
   ...
    
   julia> quadloss(ypred, ytst)
   289.7437322259235

The quadratic loss function ``quadloss`` computes :math:`(1/2n) \sum
(\hat{y} - y)^2`, i.e. half of the expected squared difference between
a predicted answer and the correct answer.  Given that y values range
from 5 to 50, `RMSE
<https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_ =
:math:`\sqrt{2\times 289.7}=24.07` is a pretty bad score.

Knet provides four functions to help train models:

================================= ==============================================================================
Function                	  Description
================================= ==============================================================================
:func:`forw(f,x) <forw>`	  returns the prediction of model f on input x
:func:`back(f,y,loss) <back>`	  computes the gradient of the parameters of f wrt the gold answers y and a loss function
:func:`update!(f) <update!>`	  updates the parameters of f using the gradients to improve loss
:func:`setp(f; kwargs...) <setp>` can be used to configure update options such as the learning rate ``lr``
================================= ==============================================================================

Using these, we can write a simple training script:

.. testcode::
   
    function train(f, x, y, loss)
        for i=1:size(x,2)
            forw(f, x[:,i])
            back(f, y[:,i], loss)
            update!(f)
        end
    end

.. testoutput::
   :hide:
      
   ...

... and train our model after setting an appropriate learning rate:

.. doctest::

   julia> setp(f, lr=0.001)
   
   julia> for i=1:100; train(f, xtrn, ytrn, quadloss); end

100 epochs of training should take a few seconds, and this time
our RMSE should be much better:

.. doctest::
   
   julia> ypred = forw(f, xtst)
   1x106 Array{Float64,2}:
   ...

   julia> quadloss(ypred,ytst)
   12.334981140829859

   julia> sqrt(2*ans)
   4.966886578296279

We can see what the model has learnt looking at the new weights:

.. doctest::

   julia> get(f,:w)
   1x13 Array{Float64,2}:
    -0.426154  0.765073  0.287288 ... -1.94362  0.837376  -3.45769

   julia> sortperm(vec(get(f,:w)))
   13-element Array{Int64,1}:
    13
     8
    ...
     9
     6

The two weights with the most negative contributions are 13 and 8.  We
can find out from `UCI
<http://archive.ics.uci.edu/ml/datasets/Housing>`_ that these are::

  13. LSTAT: % lower status of the population
  8. DIS: weighted distances to five Boston employment centres

And the two with the most positive contributions are 9 and 6::

  9. RAD: index of accessibility to radial highways 
  6. RM: average number of rooms per dwelling
      
Now, there are a lot more efficient and elegant ways to perform and
analyze a linear regression as you can find out from any decent
statistics text.  However the basic method outlined in this section
has the advantage of being easy to generalize to models that are a lot
more complicated as we will see next.

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
