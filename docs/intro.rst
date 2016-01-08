***********************
A Tutorial Introduction
***********************

We will begin by a quick tutorial on Knet.

  TODO: add intro/conclusion at all levels.

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
the lack of GPU support.  Usually, these can be safely ignored.  To
make sure everything has installed correctly, type
``Pkg.test("Knet")`` which should take a couple of minutes kicking the
tires.  If all is OK, continue with the next section, if not you can
get help at the knet-users_ mailing list.

.. _knet-users: https://groups.google.com/forum/#!forum/knet-users

Models, functions, and operators
--------------------------------
..
   @knet, compile, forw, get, primitive ops

To start using Knet, type ``using Knet`` at the Julia prompt.

.. doctest::

   julia> using Knet
   ...

In Knet, a machine learning model is defined using a special function
syntax with the ``@knet`` macro.  The following example defines a Knet
function for a simple linear regression model with 13 inputs and 1
output. You can type this definition at the Julia prompt, or you can
copy and paste it into a file which can be loaded into Julia using
``include("filename")``:

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

This looks a lot like a regular `Julia function definition`_
except for the ``@knet`` macro.  However it is important to emphasize
that the ``@knet`` macro does not define ``lin`` as a regular Julia
function or variable.  Furthermore, only a restricted set of statement
types (e.g. assignment and return statements) and operators
(e.g. ``par``, ``*`` and ``.+``) can be used in a @knet function
definition.  A full list of Knet primitive operators is given below:

.. _Julia function definition: http://julia.readthedocs.org/en/release-0.4/manual/functions>

.. _primitives-table:

   ===============================	==============================================================================
   Operator                		Description
   ===============================	==============================================================================
   :func:`par() <par>`			a parameter array, updated during training; kwargs: [#]_ ``dims, init``
   :func:`rnd() <rnd>`			a random array, updated every call; kwargs: ``dims, init``
   :func:`arr() <arr>`           	a constant array, never updated; kwargs: ``dims, init``
   :func:`dot(A,B) <dot>`        	matrix product of ``A`` and ``B``; alternative notation: ``A * B``
   :func:`add(A,B) <add>`		elementwise broadcasting [#]_ addition of arrays ``A`` and ``B``, alternative notation: ``A .+ B``
   :func:`mul(A,B) <mul>`        	elementwise broadcasting multiplication of arrays ``A`` and ``B``; alternative notation: ``A .* B``
   :func:`conv(W,X) <conv>`       	convolution [#]_ with filter ``W`` and input ``X``; kwargs: ``padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION``
   :func:`pool(X) <pool>`		pooling; kwargs: ``window=2, padding=0, stride=window, mode=CUDNN_POOLING_MAX``
   :func:`axpb(X) <axpb>`         	computes ``a*x^p+b``; kwargs: ``a=1, p=1, b=0``
   :func:`copy(X) <copy>`         	copies ``X`` to output.
   :func:`relu(X) <relu>`		rectified linear activation function: ``(x > 0 ? x : 0)``
   :func:`sigm(X) <sigm>`		sigmoid activation function: ``1/(1+exp(-x))``
   :func:`soft(X) <soft>`		softmax activation function: ``(exp xi) / (Σ exp xj)``
   :func:`tanh(X) <tanh>`		hyperbolic tangent activation function.
   ===============================	==============================================================================

.. [#] Both Julia and Knet functions accept optional `keyword
       arguments`_ Functions with keyword arguments are defined using
       a semicolon in the signature, e.g. ``plot(x, y; width=1,
       height=2)``, the semicolon is optional when the function is
       called, e.g. both ``plot(x, y, width=2)`` or ``plot(x, y;
       width=2)`` work.  Unspecified keyword arguments take their
       default values specified in the function definition.

.. [#] `Broadcasting operations`_ are element-by-element binary
       operations on arrays of possibly different sizes, such as
       adding a vector to each column of a matrix.  They expand
       singleton dimensions in array arguments to match the
       corresponding dimension in the other array without using extra
       memory, and apply the given function elementwise.

.. [#] For detailed information about convolution and pooling, please
       see the documentation for CUDNN_ and `CUDNN.jl`_.

.. _keyword arguments: http://julia.readthedocs.org/en/release-0.4/manual/functions/#keyword-arguments
.. _Broadcasting operations: http://julia.readthedocs.org/en/release-0.4/manual/arrays/#broadcasting
.. _CUDNN: https://developer.nvidia.com/cudnn
.. _CUDNN.jl: https://github.com/JuliaGPU/CUDNN.jl

In order to turn ``lin`` into a machine learning model that can be
trained with examples and used for predictions, we need to compile it:

.. doctest:: :hide:

    julia> setseed(42);

.. doctest::

    julia> f = compile(:lin)	# The colon before lin is required
    ...

..
   This defines ``f`` as an actual model (model or Net?) that we can
   train and use for predictions (repeated).  Note that the colon
   character preceding the name of our Knet function is required in the
   compile expression.  (TODO: can we get rid of the colon with a macro?)
   (TODO: The motivation behind this two step process, first defining a
   Knet function then compiling it into a model, will become more clear
   when we introduce compile time parameters.)

To test our model let's give it some input.  ``w`` is a :math:`1\times
13` row vector, so the input ``x`` should be a :math:`13\times 1`
column vector:

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

We can query the model and see its parameters using ``get``:
      
.. doctest::

    julia> get(f,:w)
    1x13 Array{Float64,2}:
     -0.556027  -0.444383  0.0271553 ... 1.08238  0.187028  0.518149

    julia> get(f,:b)
    1x1 Array{Float64,2}:
     1.49138
    
Note that we need to escape Knet variable names using the `colon
character`_
just like we did for ``:lin`` when compiling.  We can also look at the
input with ``get(f,:x)``, reexamine the output using the special
``:return`` symbol with ``get(f,:return)``.  In fact using ``get``, we
can confirm that our model gives us the same answer as an equivalent
Julia expression:

.. doctest::     

    julia> get(f,:w) * get(f,:x) .+ get(f,:b)
    1x1 Array{Float64,2}:
     -1.00532

.. _colon character: http://julia.readthedocs.org/en/release-0.4/manual/metaprogramming#symbols

..
   Also note that ``lin`` is not defined as a regular Julia function or
   variable.

   .. doctest

      julia> lin(5)
      ERROR: UndefVarError: lin not defined

..
   So far it looks like all Knet gave us is a very complicated way to
   define a very simple function.  So why would anybody bother defining a
   @knet function with all the syntactic restrictions, limited number of
   operators, need for compilation etc.?

Training a model
----------------
..
   quadloss, back, update!, setp, update options

OK, we can define functions using Knet but why should we bother?  What
makes a Knet model different from an ordinary function is that Knet
models are `differentiable programs`.  This means that for a given
input not only can they compute an output, but they can also compute
which way their parameters should be modified to approach some desired
output.  If we have some input-output data that comes from an unknown
function, we can `train` a Knet model to look like this unknown
function by manipulating its parameters.

Let us download the `Housing dataset`_ from the `UCI Machine Learning
Repository`_ to train our ``lin`` model:

.. _Housing dataset: http://archive.ics.uci.edu/ml/datasets/Housing

.. _UCI Machine Learning Repository: http://archive.ics.uci.edu/ml/datasets.html

.. doctest::
   
   julia> using Requests
   julia> url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data";
   julia> data = readdlm(get(url).data)'  # Don't forget the final apostrophe to transpose
   14x506 Array{Float64,2}:...

The dataset has housing related information for 506 neighborhoods in
Boston, each with 14 attributes.  The last attribute is the median
house price to be predicted, so let's separate it:

.. doctest::
   
   julia> x = data[1:13,:]
   13x506 Array{Float64,2}:...
   julia> y = data[14,:]
   1x506 Array{Float64,2}:...

You may have noticed that the input attributes have very different
ranges.  It is usually a good idea to normalize them:

.. doctest::

   julia> x = (x .- mean(x,2)) ./ std(x,2);

It is also a good idea to split our dataset into train and test
portions so we can estimate how well our model will do on unseen data:

.. doctest::

   julia> r = randperm(size(x,2));
   julia> xtrn=x[:,r[1:400]];
   julia> ytrn=y[:,r[1:400]];
   julia> xtst=x[:,r[401:end]];
   julia> ytst=y[:,r[401:end]];
    
Let's see how well our randomly initialized model does before
training:

.. doctest::

   julia> ypred = forw(f, xtst)
   1x106 Array{Float64,2}:...
   julia> quadloss(ypred, ytst)
   289.7437322259235

The quadratic loss function ``quadloss`` computes :math:`(1/2n) \sum
(\hat{y} - y)^2`, i.e. half of the mean squared difference between a
predicted answer :math:`\hat{y}` and the desired answer :math:`y`.
Given that :math:`y` values range from 5 to 50, a `root mean squared
error (RMSE)`_ of :math:`\sqrt{2\times 289.7}=24.07` is a pretty bad
score.

.. _root mean squared error (RMSE): https://en.wikipedia.org/wiki/Root-mean-square_deviation

We would like to minimize this loss which should get the predicted
answer closer to the desired answer.  To do this we first compute the
loss gradient for the parameters of ``f`` -- this is the direction in
parameter space that maximally increase the loss.  Then we move the
parameters in the opposite direction.

Knet provides three functions to help train models:

================================= ==============================================================================
Function                	  Description
================================= ==============================================================================
:func:`forw(f,x) <forw>`	  returns the prediction of model ``f`` on input ``x``
:func:`back(f,y,lossfn) <back>`	  computes the loss gradients of ``f`` parameters based on the desired output ``y`` and a loss function ``lossfn``
:func:`update!(f) <update!>`	  updates the parameters of ``f`` using the gradients computed by ``back`` to reduce loss
================================= ==============================================================================

..
   TODO: remove the ! from update! ?
   TODO: have an objective function instead of a loss function?

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

Here is the sequence of events that take place during training:

* The ``for`` loop grabs training instances one by one.
* ``forw`` computes the prediction for the i'th instance.  This is required for the next step.
* ``back`` computes the loss gradient ``dw`` for each parameter ``w`` for the i'th instance.
* ``update!`` subtracts (a function of) ``dw`` from ``w`` to reduce the loss for each parameter ``w``.

We can manipulate how exactly ``update!`` behaves by setting some
training options like the learning rate ``lr``.  I'll explain the
mathematical motivation later, but algorithmically these training
options manipulate the ``dw`` array (sometimes using an auxiliary
array ``dw2``) before the subtraction to improve the loss faster.
Here is a list of training options supported by Knet and how they
manipulate ``dw``:

.. _training-options-table:

=============================== ==============================================================================
Option	                	Description
=============================== ==============================================================================
``lr``				Learning rate: ``dw *= lr``
``l1reg``			L1 regularization: ``dw += l1reg * sign(w)``
``l2reg``			L2 regularization: ``dw += l2reg * w``
``adagrad``			Adagrad (boolean): ``dw2 += dw .* dw; dw = dw ./ (1e-8 + sqrt(dw2))``
``momentum``			Momentum: ``dw += momentum * dw2; dw2 = dw``
``nesterov``			Nesterov: ``dw2 = nesterov * dw2 + dw; dw += nesterov * dw2``
=============================== ==============================================================================


We can set these training options for individual parameters using
e.g. ``setp(f, :w; lr=0.001)``, or for the whole model using ``setp(f;
lr=0.001)``.  Let's set the learning rate to 0.001 and train the model
for 100 epochs:

.. doctest::

   julia> setp(f; lr=0.001)
   julia> for i=1:100; train(f, xtrn, ytrn, quadloss); end

This should take a few seconds, and this time our RMSE should be much
better:

.. doctest::
   
   julia> ypred = forw(f, xtst)
   1x106 Array{Float64,2}:...
   julia> quadloss(ypred,ytst)
   12.334981140829859
   julia> sqrt(2*ans)
   4.966886578296279

We can see what the model has learnt looking at the new weights:

.. doctest::

   julia> get(f,:w)
   1x13 Array{Float64,2}:
    -0.426154  0.765073  0.287288 ... -1.94362  0.837376  -3.45769

..
   julia> println(sortperm(vec(get(f,:w))))
   [13,8,11,5,10,1,7,3,2,4,12,9,6]

The two weights with the most negative contributions are 13 and 8.  We
can find out from UCI_ that these are::

  13. LSTAT: % lower status of the population
   8. DIS: weighted distances to five Boston employment centres

And the two with the most positive contributions are 9 and 6::

   9. RAD: index of accessibility to radial highways 
   6. RM: average number of rooms per dwelling
      
.. _UCI: http://archive.ics.uci.edu/ml/datasets/Housing

Now, there are a lot more efficient and elegant ways to perform and
analyze a linear regression as you can find out from any decent
statistics text.  However the basic method outlined in this section
has the advantage of being easy to generalize to models that are a lot
more complicated as we will see next.

Defining new operators
----------------------
..
   @knet as op, kwargs for @knet functions,
   function options (f=:relu).  splat.
   lenet example, fast enough on cpu?

The key to controlling complexity in computer languages is
*abstraction*.  Abstraction is the ability to name compound structures
built from primitive parts, so they too can be used as primitives.  In
Knet we do this by using @knet functions not as models, but as new
operators inside other @knet functions.

We will use the LeNet_ convolutional neural network model to
illustrate this.  Here is the LeNet model [#]_ defined only using
primitives [#]_ from the :ref:`Knet primitives table
<primitives-table>`:

.. testcode::

    @knet function lenet1(x)    # dims=(28,28,1,N)
        w1 = par(init=Xavier(),   dims=(5,5,1,20))
        c1 = conv(w1,x)         # dims=(24,24,20,N)
        b1 = par(init=Constant(0),dims=(1,1,20,1))
        a1 = add(b1,c1)
        r1 = relu(a1)
        p1 = pool(r1)           # dims=(12,12,20,N)

        w2 = par(init=Xavier(),   dims=(5,5,20,50))
        c2 = conv(w2,p1)        # dims=(8,8,50,N)
        b2 = par(init=Constant(0),dims=(1,1,50,1))
        a2 = add(b2,c2)
        r2 = relu(a2)
        p2 = pool(r2)           # dims=(4,4,50,N)

        w3 = par(init=Xavier(),   dims=(500,800))
        d3 = dot(w3,p2)         # dims=(500,N)
        b3 = par(init=Constant(0),dims=(500,1))
        a3 = add(b3,d3)
        r3 = relu(a3)

        w4 = par(init=Xavier(),   dims=(10,500))
        d4 = dot(w4,r3)         # dims=(10,N)
        b4 = par(init=Constant(0),dims=(10,1))
        a4 = add(b4,d4)
        return soft(a4)         # dims=(10,N)
    end

.. testoutput:: :hide:

   ...

.. [#] This definition closely follows the Caffe_ implementation.

.. _Caffe: http://caffe.berkeleyvision.org/gathered/examples/mnist.html

.. [#] ``Xavier()`` and ``Constant(0)`` are random number
       distributions that can be used to initialize Knet parameters,
       they will be covered in detail later.

At 22 lines long, this model looks a lot more complicated than our
linear regression model.  Compared to state of the art object
recognition models however, it is still tiny.  You would not want to
code a model like GoogLeNet_ using these primitives.

.. _GoogLeNet: http://arxiv.org/abs/1409.4842

If you look closely, the LeNet model has two convolution-pooling
layers, a fully connected relu layer and a final softmax output layer
(separated by blank lines).  Wouldn't it be nice to say just *that*::

    @knet function lenet2(x)
        a = conv_pool_layer(x)
        b = conv_pool_layer(a)
        c = relu_layer(b)
        return softmax_layer(c)
    end
    
``lenet2`` is a lot more readable than ``lenet1``.  But before we can
use this definition, we have to solve two problems:

* ``conv_pool_layer`` etc. are not primitive operators, we need a way to add them to the Knet language.
* Each layer has some attributes, like ``init`` and ``dims``, that we need to be able to configure.

Knet solves the first problem by allowing @knet functions to be used
as operators as well as models.  For example::

    @knet function conv_pool_layer(x)
        w = par(init=Xavier(), dims=(5,5,1,20))
        c = conv(w,x)
        b = par(init=Constant(0), dims=(1,1,20,1))
        a = add(b,c)
        r = relu(a)
        return pool(r)
    end

With this definition, the the first ``a = conv_pool_layer(x)``
operation in ``lenet2`` will work exactly as we want, but not the
second.

This brings us to the second problem, layer configuration.  It would
be nice not to hard code numbers like ``(5,5,1,20)`` in the definition
of a new operation like ``conv_pool_layer``.  Making these numbers
configurable would make such operations more reusable across models.
Even within the same model, you may want to use the same layer type in
more than one configuration.  For example in ``lenet2`` there is no
way to distinguish the two ``conv_pool_layer`` operations, but looking
at ``lenet1`` we clearly want them to do different things.

Knet solves the layer configuration problem using `keyword
arguments`_.  Slightly modifying the definition of
``conv_pool_layer``::

    @knet function conv_pool_layer(x; winit=Xavier(), wdims=(5,5,1,20), binit=Constant(0), bdims=(1,1,20,1))
        w = par(init=winit, dims=wdims)
        c = conv(w,x)
        b = par(init=binit, dims=bdims)
        a = add(b,c)
        r = relu(a)
        return pool(r)
    end

would allow us to distinguish the two ``conv_pool_layer`` operations:

.. testcode::

    @knet function lenet3(x)
        a = conv_pool_layer(x; wdims=(5,5,1,20),  bdims=(1,1,20,1))
        b = conv_pool_layer(a; wdims=(5,5,20,50), bdims=(1,1,50,1))
        c = relu_layer(b; wdims=(500,800), bdims=(500,1))
        return softmax_layer(c; wdims=(10,500), bdims=(10,1))
    end

.. testoutput:: :hide:

   ...

In fact, we can use keyword arguments to define a ``generic_layer``
that contains the shared code for all our layers:

.. testcode::

    @knet function generic_layer(x; f1=:relu, f2=:dot, winit=Xavier(), binit=Constant(0), wdims=(), bdims=())
        w = par(init=winit, dims=wdims)
        y = f2(w,x)
        b = par(init=binit, dims=bdims)
        z = add(b,y)
        return f1(z)
    end

.. testoutput:: :hide:

   ...

Note that in this example we are not only making initialization
parameters like ``winit`` and ``binit`` configurable, we are also
making operators like ``relu`` and ``dot`` configurable (note the
colons in the first line).  This generic layer will allow us to define
many layer types easily:

.. testcode::

    @knet function conv_pool_layer(x; o...)
        y = generic_layer(x; o..., f2=:conv)
        return pool(y)
    end

    @knet function relu_layer(x; o...)
        return generic_layer(x; o...)
    end

    @knet function softmax_layer(x; o...)
        return generic_layer(x; o..., f1=:soft)
    end

.. testoutput:: :hide:

   ...

The ``...`` notation in the function definitions and calls above is
Julia's `slurp and splat operator`_.  Its usage here basically says
that whatever keyword arguments you pass the ``relu_layer``, for
example, it will pass them down to the ``generic_layer``.

.. _slurp and splat operator: http://julia.readthedocs.org/en/release-0.4/manual/faq/?highlight=splat#what-does-the-operator-do

Using new operators and keyword arguments, not only did we cut the
amount of code in half, we made the definition of LeNet a lot more
readable and gained a bunch of reusable operators to boot.  I am sure
you can think of more clever ways to define LeNet and similar models
using the power of abstraction.  To see some example reusable
operators take a look at `kfun.jl`_.

.. _kfun.jl: https://github.com/denizyuret/Knet.jl/blob/master/src/kfun.jl

..
   TODO: repeat, zero sizes and size inference, keyword args to compile(), rgen distributions.

Minibatches
-----------
..
   minibatch, softloss, zeroone

We will use the LeNet model to classify hand-written digits from the
MNIST_ dataset.  The following downloads the MNIST data:

.. _LeNet: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
.. _MNIST: http://yann.lecun.com/exdb/mnist

.. doctest::

    julia> include(Pkg.dir("Knet/examples/mnist.jl"))
    ...

Once loaded, the data is available as multi-dimensional Julia arrays
in the MNIST module:

.. doctest::

    julia> MNIST.xtrn
    28x28x1x60000 Array{Float32,4}:...
    julia> MNIST.ytrn
    10x60000 Array{Float32,2}:...
    julia> MNIST.xtst
    28x28x1x10000 Array{Float32,4}:...
    julia> MNIST.ytst
    10x10000 Array{Float32,2}:...

We have 60000 training and 10000 testing examples.  Each input x is a
28x28x1 image, where the first two numbers represent the width and
height in pixels, the third number is the number of channels (which is
1 for grayscale images, 3 for RGB images etc.)  The pixel values have
been normalized to :math:`[0,1]`.  Each output y is a ten-dimensional
one-hot vector (a vector that has a single non-zero component)
indicating the correct class for a given image.

This is a much larger dataset than Housing.  For computational
efficiency, it is not advisable to use these examples one at a time
during training like we did in the regression example.  The following
will split the data into groups of 100 examples called minibatches:

.. doctest::

    julia> batchsize=100;
    julia> trn = minibatch(MNIST.xtrn, MNIST.ytrn, batchsize)
    600-element Array{Any,1}:...
    julia> tst = minibatch(MNIST.xtst, MNIST.ytst, batchsize)
    100-element Array{Any,1}:...

Each element of ``trn`` and ``tst`` is an x, y pair that contains 100
examples:

.. doctest::
 
    julia> trn[1]
    (
    28x28x1x100 Array{Float32,4}:
    ...
    10x100 Array{Float32,2}:
    ...)

Here are some simple train and test scripts that use this minibatched
data:

.. testcode::

    function train(f, data, loss)
        for (x,y) in data
            forw(f, x)
            back(f, y, loss)
            update!(f)
        end
    end

    function test(f, data, loss)
        sumloss = numloss = 0
        for (x,ygold) in data
            ypred = forw(f, x)
            sumloss += loss(ypred, ygold)
            numloss += 1
        end
        sumloss / numloss
    end

.. testoutput::
   :hide:
      
   ...

OK, now we are ready to train the LeNet model with the minibatched
MNIST data:

.. doctest::

   julia> net = compile(:lenet3);
   julia> setp(net; lr=0.1);
   julia> train(net, trn, softloss);
   julia> test(net, tst, zeroone)
   0.0226

We compile the model and set the learning rate to 0.1, which works
well for this example.  We also see two new loss functions in this
example: ``softloss`` computes the *cross entropy loss*,
:math:`E(p\log\hat{p})`, commonly used for training classification
models and ``zeroone`` computes the *zero-one loss* which is the ratio
of predictions that were wrong.

  TODO: give some timing information, that was our motivation here.
  possibly show the source for minibatch.  point to the iterable
  interface and knet examples for less memory waste on bigger datasets. 

After one epoch of training we get 2.26% test error [#]_.  You should
be able to get this down to 0.8% in about 30 epochs of training.  See
the MNIST_ web page for some benchmark results on this dataset.

.. [#] Your results may be slightly different if you are using a GPU
       machine because some of the convolution operations are non-deterministic.   

RNNs
----
..
   read-before-write, simple rnn, lstm

In this section we will see how to implement *recurrent neural
networks* (RNNs) in Knet.  All local variables in Knet functions are
`static variables`_, i.e. their values are preserved between calls
unless otherwise specified.  It turns out this is the only language
feature you need to define RNNs.  Here is a simple example::

    @knet function rnn1(x; hsize=100, xsize=50)
        a = par(init=Xavier(), dims=(hsize, xsize))
        b = par(init=Xavier(), dims=(hsize, hsize))
        c = par(init=Constant(0), dims=(hsize, 1))
        d = a * x .+ b * h .+ c
        h = relu(d)
    end

..

  TODO: would be nice to use 0 for xsize at this point.  Also this is
  the second time we are using Xavier etc without much explanation.

.. _static variables: https://en.wikipedia.org/wiki/Static_variable

Notice anything strange?  The first three lines define three model
parameters.  Then the fourth line sets ``d`` to a linear combination
of the input ``x`` and the hidden state ``h``.  But ``h`` hasn't been
defined yet.  Exactly!  Having read-before-write variables is the only
thing that distinguishes an RNN @knet function from feed-forward
models like LeNet.

The way Knet handles read-before-write variables is by initializing
them to 0 arrays before any input is processed, then preserving the
values between the calls.  Thus during the first call in the above
example, ``h`` would start as 0, ``d`` would be set to ``a * x .+ c``,
which in turn would cause ``h`` to get set to ``relu(a * x .+ c)``.
During the second call, this value of ``h`` would be remembered and
used, thus making the value of ``h`` at time t dependent on
its value at time t-1.

Sequences
---------
..
   how to represent sequence data? karpathy example?  need generator.
   Karpathy Technical: Lets train a 2-layer LSTM with 512 hidden nodes
   (approx. 3.5 million parameters), and with dropout of 0.5 after
   each layer. We'll train with batches of 100 examples and truncated
   backpropagation through time of length 100 characters. With these
   settings one batch on a TITAN Z GPU takes about 0.46 seconds (this
   can be cut in half with 50 character BPTT at negligible cost in
   performance). Without further ado, lets see a sample from the RNN:

   In RNNs past inputs effect future outputs.  Thus they are typically
   used to process sequences, such as speech or text data.

.. _karpathy: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

.. _shakespeare: http://www.gutenberg.org/files/100/100.txt


Conditionals
------------

There are cases where you want to execute parts of a model
*conditionally*, e.g. only during training, or only during some parts
of the input in sequence models.  Knet supports the use of *runtime
conditions* for this purpose.

Dropout
-------
..
   if-else, runtime conditions (kwargs for forw), dropout
   lenet with dropout?  fast enough for cpu?
   lenet is not a good example for dropout does not converge very fast.  dropout may not be
   a good motivator for conditionals: there are other ways to
   implement dropout?, s2c, s2s models may be better?
   lenet with drop=0.4 drop1=0.0 adaptive lr with decay=0.9 gets 0.5%
   (min .0045) in 100 epochs.  with fixed lr=0.1 gets <0.5% in 50
   epochs so no need for the adaptive lr. hmm trying to replicate, 50
   is not enough.
   this should probably come after rnns and sequences.
   could make this a dropout section and have a different conditional
   section. as a dropout section it doesn't need to be in the
   tutorial.  if this is going to be its own section, put more about
   the theory, the alternatives, other types of noise introduction
   papers.


If you keep training the LeNet model on MNIST for about 30 epochs you
will observe that the training error drops to zero but the test error
hovers around 0.8%::

    for epoch=1:100
        train(net, trn, softloss)
        println((epoch, test(net, trn, zeroone), test(net, tst, zeroone)))
    end

    (1,0.020466666666666505,0.024799999999999996)
    (2,0.013649999999999905,0.01820000000000001)
    ...
    (29,0.0,0.008100000000000003)
    (30,0.0,0.008000000000000004)

This is called *overfitting*.  The model has memorized the training
set, but does not generalize equally well to the test set.  There are
many ways to reduce overfitting: more training data, a smaller model
with fewer parameters, regularization (remember the ``l1reg`` and
``l2reg`` from the :ref:`table of training options
<training-options-table>`), and early stopping can all help and will
be covered elsewhere (TODO).  In this section we will look at a more
recent technique called dropout_.

.. _dropout: http://jmlr.org/papers/v15/srivastava14a.html

For each ``forw`` call during training, dropout replaces a certain
percentage of the output of an operation with zeros, and scales the
rest to keep the total output the same.  During testing 
