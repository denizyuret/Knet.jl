***********************
A Tutorial Introduction
***********************

We will begin by a quick tutorial on Knet, going over the essential
tools for defining, training, and evaluating real machine learning
models.  The goal is to get you to the point where you can create your
own models and apply machine learning to your own problems as quickly
as possible.  So some of the details and exceptions will be skipped
for now.  No prior knowledge of machine learning or Julia is
necessary, but general programming experience will be assumed.  It
would be best if you follow along with the examples on your computer,
please see :ref:`installation-section` for setup instructions.


Models, functions, and operators
--------------------------------
.. @knet, compile, forw, get

In this section, we will create our first Knet model, learn how to
peek into it, and how to make a prediction.  To start using Knet, type
``using Knet`` at the Julia prompt::

   julia> using Knet

.. testsetup::

   using Knet
   setseed(42);

In Knet, a machine learning model is defined using a special function
syntax with the ``@knet`` macro.  The following example defines a
@knet function for a simple linear regression model with 13 inputs and
a single output. You can type this definition at the Julia prompt, or
you can copy and paste it into a file which can be loaded into Julia
using ``include("filename")``:

.. testcode::

    @knet function lin(x)
        w = par(init=randn(1,13))
        b = par(init=randn(1,1))
        return w * x .+ b
    end

.. testoutput:: :hide:

   ...

.. _randn: http://julia.readthedocs.org/en/release-0.4/stdlib/numbers/#Base.randn
.. _Julia function: http://julia.readthedocs.org/en/release-0.4/manual/functions
.. _variable: http://julia.readthedocs.org/en/release-0.4/manual/variables

In this definition:

- ``@knet`` indicates that ``lin`` is a special Knet function, it does
  not get defined as a regular `Julia function`_ or variable_.
- ``x`` is the only input argument.
- ``w`` and ``b`` are model parameters as indicated by the ``par``
  constructor.
- ``init`` is a keyword argument to ``par`` describing how the
  parameter should be initialized.  It can take an array or one of the
  supported :ref:`random distributions <rgen-table>`.
- :func:`randn(dims) <randn>` is a Julia function that returns an
  array of size ``dims`` filled with random numbers from the standard
  normal distribution.
- The final ``return`` statement specifies the output of the Knet
  function, where ``*`` denotes matrix product and ``.+`` denotes
  elementwise addition.  Only a restricted set of :ref:`operators
  <primitives-table>` (e.g. ``*`` and ``.+``) and statement types
  (e.g. assignment and return statements) can be used in a @knet
  function definition.

In order to turn ``lin`` into a machine learning model that can be
trained with examples and used for predictions, we need to compile it:

.. doctest::

    julia> f = compile(:lin)	# The colon before lin is required
    ...

To test our model let's give it some input.  ``w`` is a :math:`1\times
13` row vector, so the input ``x`` should be a :math:`13\times 1`
column vector:

.. doctest::

    julia> x = randn(13,1)
    13x1 Array{Float64,2}:...

To obtain the prediction of model ``f`` on input ``x`` we use the
``forw`` function, which basically calculates ``w * x .+ b``:

.. doctest::     
    
    julia> forw(f,x)
    1x1 Array{Float64,2}:
     -1.00532

We can query the model and see its parameters using ``get``:
      
.. doctest::

    julia> get(f,:w)		# The colon before w is required
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


Training a model
----------------
..
   quadloss, back, update!, setp, update options

OK, so we can define functions using Knet but why should we bother?
The thing that makes a Knet model different from an ordinary function
is that Knet models are **differentiable programs**.  This means that
for a given input not only can they compute an output, but they can
also compute which way their parameters should be modified to approach
some desired output.  If we have some input-output data that comes
from an unknown function, we can train a Knet model to look like this
unknown function by manipulating its parameters.

.. _Housing: http://archive.ics.uci.edu/ml/datasets/Housing
.. _UCI Machine Learning Repository: http://archive.ics.uci.edu/ml/datasets.html

We will use the Housing_ dataset from the `UCI Machine Learning
Repository`_ to train our ``lin`` model.  The dataset has housing
related information for 506 neighborhoods in Boston, each with 14
attributes.  Here are the first 3 entries::

    0.00632  18.00   2.310  0  0.5380  6.5750  65.20  4.0900   1  296.0  15.30 396.90   4.98  24.00
    0.02731   0.00   7.070  0  0.4690  6.4210  78.90  4.9671   2  242.0  17.80 396.90   9.14  21.60
    0.02729   0.00   7.070  0  0.4690  7.1850  61.10  4.9671   2  242.0  17.80 392.83   4.03  34.70
    ...

.. _Requests: https://github.com/JuliaWeb/Requests.jl
.. _readdlm: http://julia.readthedocs.org/en/release-0.4/stdlib/io-network/#Base.readdlm

Let's download the dataset using Requests_, a Julia module that
enables downloading files from the internet using the :func:`get`
function and :func:`readdlm <readdlm>`, a function which turns space
or tab delimited data into a Julia array.  If for some reason this
does not work, you can download the data file from the given URL by
other means and run ``readdlm("housing.data")`` instead::

   julia> using Requests
   julia> url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data";
   julia> data = readdlm(get(url).data)'  # Don't forget the final apostrophe to transpose data
   14x506 Array{Float64,2}:...

.. doctest:: :hide:
   
   julia> data = readdlm(Pkg.dir("Knet/data/housing.data"))';
   
The resulting ``data`` matrix should have 506 columns representing
neighborhoods, and 14 rows representing the attributes.  The last
attribute is the median house price to be predicted, so let's separate
it:

.. doctest::
   
   julia> x = data[1:13,:]
   13x506 Array{Float64,2}:...
   julia> y = data[14,:]
   1x506 Array{Float64,2}:...

.. _Julia's array indexing: http://julia.readthedocs.org/en/release-0.4/manual/arrays/#indexing

Here we are using `Julia's array indexing`_ notation to split the data
array into input ``x`` and output ``y``.  Inside the square brackets
``1:13`` means grab the rows 1 through 13, and the ``:`` character by
itself means grab all the columns.

You may have noticed that the input attributes have very different
ranges.  It is usually a good idea to normalize them by subtracting
the mean and dividing by the standard deviation:

.. doctest::

   julia> x = (x .- mean(x,2)) ./ std(x,2);

The :func:`mean` and :func:`std` functions compute the mean and
standard deviation of ``x``.  Their optional second argument gives the
dimensions to sum over, so ``mean(x)`` gives us the mean of the whole
array, ``mean(x,1)`` gives the mean of each column, and ``mean(x,2)``
gives us the mean of each row.

It is also a good idea to split our dataset into training and test
subsets so we can estimate how well our model will do on unseen data.

.. doctest::

   julia> n = size(x,2);
   julia> r = randperm(n);
   julia> xtrn=x[:,r[1:400]];
   julia> ytrn=y[:,r[1:400]];
   julia> xtst=x[:,r[401:end]];
   julia> ytst=y[:,r[401:end]];
    
``n`` is set to the number of instances (columns) and ``r`` is set to
:func:`randperm(n) <randperm>` which gives a random permutation of
integers :math:`1\ldots n`.  The first 400 indices in ``r`` will be
used for training, and the last 106 for testing.

Let's see how well our randomly initialized model does before
training:

.. doctest::

   julia> ypred = forw(f, xtst)
   1x106 Array{Float64,2}:...
   julia> quadloss(ypred, ytst)
   289.7437322259235

The quadratic loss function :func:`quadloss` computes :math:`(1/2n)
\sum (\hat{y} - y)^2`, i.e. half of the mean squared difference
between a predicted answer :math:`\hat{y}` and the desired answer
:math:`y`.  Given that :math:`y` values range from 5 to 50, an RMSD_
of :math:`\sqrt{2\times 289.7}=24.07` is a pretty bad score.

.. _RMSD: https://en.wikipedia.org/wiki/Root-mean-square_deviation

We would like to minimize this loss which should get the predicted
answers closer to the desired answers.  To do this we first compute
the loss gradient for the parameters of ``f`` -- this is the direction
in parameter space that maximally increases the loss.  Then we move the
parameters in the opposite direction.  Here is a simple function that
performs these steps:

..
   TODO: remove the ! from update! ?
   TODO: have an objective function instead of a loss function?

.. Using these, we can write a simple training script:

.. testcode::
   
    function train(f, x, y)
        for i=1:size(x,2)
            forw(f, x[:,i])
            back(f, y[:,i], quadloss)
            update!(f)
        end
    end

.. testoutput::
   :hide:
      
   ...


* The ``for`` loop grabs training instances one by one.
* ``forw`` computes the prediction for the i'th instance.  This is required for the next step.
* ``back`` computes the loss gradient ``dw`` for each parameter ``w`` for the i'th instance.
* ``update!`` subtracts (a function of) ``dw`` from ``w`` to reduce the loss for each parameter ``w``.


.. We can set these training options for individual parameters using
.. e.g. ``setp(f, :w; lr=0.001)``, or for the whole model using ``setp(f;
.. lr=0.001)``.  

Before training, it is important to set a good learning rate.  The
learning rate controls how large the update steps are going to be: too
small and you'd wait for a long time, too large and ``train`` may
never converge.  The :func:`setp` function is used to set training
options like the learning rate.  Let's set the learning rate to 0.001
and train the model for 100 epochs (i.e. 100 passes over the dataset):

.. doctest::

   julia> setp(f, lr=0.001)
   julia> for i=1:100; train(f, xtrn, ytrn); end

This should take a few seconds, and this time our RMSD should be much
better:

.. doctest::
   
   julia> ypred = forw(f, xtst)
   1x106 Array{Float64,2}:...
   julia> quadloss(ypred,ytst)
   12.3349...
   julia> sqrt(2*ans)
   4.9668...

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
**abstraction**.  Abstraction is the ability to name compound structures
built from primitive parts, so they too can be used as primitives.  In
Knet we do this by using @knet functions not as models, but as new
operators inside other @knet functions.

To illustrate this, we will use the LeNet_ convolutional neural
network model designed to recognize handwritten digits.  Here is the
LeNet model defined only using the :ref:`primitive operators of Knet
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

.. _GoogLeNet: http://arxiv.org/abs/1409.4842

.. .. _Caffe: http://caffe.berkeleyvision.org/gathered/examples/mnist.html

.. .. [#] This definition closely follows the Caffe_ implementation.

.. In our first model ``lin``, we had specified model parameters by
.. passing random arrays to the ``init`` argument.  LeNet uses a
.. different alternative, the parameters are specified by indicating
.. their size with the ``dims`` argument and random distributions
.. (``Xavier()`` and ``Constant(0)``) with the ``init`` argument.

Don't worry about the details of the model if you don't know much
about neural nets.  At 22 lines long, this model looks a lot more
complicated than our linear regression model.  Compared to state of
the art image processing models however, it is still tiny.  You
would not want to code a state-of-the-art model like GoogLeNet_ using
these primitives.

If you are familiar with neural nets, and peruse the :ref:`Knet
primitives table <primitives-table>`, you can see that the model has
two convolution-pooling layers (commonly used in image processing), a
fully connected relu layer and a final softmax output layer (I
separated them by blank lines to help).  Wouldn't it be nice to say
just *that*::

    @knet function lenet2(x)
        a = conv_pool_layer(x)
        b = conv_pool_layer(a)
        c = relu_layer(b)
        return softmax_layer(c)
    end
    
``lenet2`` is a lot more readable than ``lenet1``.  But before we can
use this definition, we have to solve two problems:

* ``conv_pool_layer`` etc. are not primitive operators, we need a way to add them to Knet.
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
second (it has different dimensions).

This brings us to the second problem, layer configuration.  It would
be nice not to hard-code numbers like ``(5,5,1,20)`` in the definition
of a new operation like ``conv_pool_layer``.  Making these numbers
configurable would make such operations more reusable across models.
Even within the same model, you may want to use the same layer type in
more than one configuration.  For example in ``lenet2`` there is no
way to distinguish the two ``conv_pool_layer`` operations, but looking
at ``lenet1`` we clearly want them to do different things.

.. _keyword arguments: http://julia.readthedocs.org/en/release-0.4/manual/functions/#keyword-arguments
.. _three dots: http://julia.readthedocs.org/en/release-0.4/manual/faq/?highlight=splat#what-does-the-operator-do

Knet solves the layer configuration problem using `keyword
arguments`_.  Knet functions borrow the keyword argument syntax from
Julia, and we will be using them in many contexts, so a brief aside is
in order: Keyword arguments are identified by name instead of
position, and they can be passed in any order (or not passed at all)
following regular (positional) arguments.  In fact we have already
seen examples: ``dims`` and ``init`` are keyword arguments for ``par``
(which has no regular arguments).  Functions with keyword arguments are
defined using a semicolon in the signature, e.g. ``function plot(x, y;
width=1, height=2)``.  The semicolon is optional when the function is
called, e.g. both ``plot(x, y, width=2)`` or ``plot(x, y; width=2)``
work.  Unspecified keyword arguments take their default values
specified in the function definition.  Extra keyword arguments can be
collected using `three dots`_ in the function definition: ``function
plot(x, y; width=1, height=2, o...)``, and passed in function calls:
``plot(x, y; o...)``.

Here is a configurable version of ``conv_pool_layer`` using keyword
arguments::

    @knet function conv_pool_layer(x; winit=Xavier(), wdims=(), binit=Constant(0), bdims=())
        w = par(init=winit, dims=wdims)
        c = conv(w,x)
        b = par(init=binit, dims=bdims)
        a = add(b,c)
        r = relu(a)
        return pool(r)
    end

This allows us to distinguish the two ``conv_pool_layer`` operations:

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

    @knet function generic_layer(x; f1=:dot, f2=:relu, winit=Xavier(), binit=Constant(0), wdims=(), bdims=())
        w = par(init=winit, dims=wdims)
        y = f1(w,x)
        b = par(init=binit, dims=bdims)
        z = add(b,y)
        return f2(z)
    end

.. testoutput:: :hide:

   ...

Note that in this example we are not only making initialization
parameters like ``winit`` and ``binit`` configurable, we are also
making internal operators like ``relu`` and ``dot`` configurable
(their names need to be escaped with colons when passed as keyword
arguments).  This generic layer will allow us to define many layer
types easily:

.. testcode::

    @knet function conv_pool_layer(x; o...)
        y = generic_layer(x; o..., f1=:conv, f2=:relu)
        return pool(y)
    end

    @knet function relu_layer(x; o...)
        return generic_layer(x; o..., f1=:dot, f2=:relu)
    end

    @knet function softmax_layer(x; o...)
        return generic_layer(x; o..., f1=:dot, f2=:soft)
    end

.. testoutput:: :hide:

   ...

TODO: we need to introduce size inference here, otherwise they won't
understand kfun.jl.

.. _kfun.jl: https://github.com/denizyuret/Knet.jl/blob/master/src/kfun.jl

Using new operators and keyword arguments, not only did we cut the
amount of code in half, we made the definition of LeNet a lot more
readable and gained a bunch of reusable operators to boot.  I am sure
you can think of more clever ways to define LeNet and similar models
using the power of abstraction.  To see some example reusable
operators take a look at the :ref:`Knet compound operators
<compounds-table>` table and see their definitions in `kfun.jl`_.


Training with minibatches
-------------------------
..
   minibatch, softloss, zeroone

We will use the LeNet model to classify hand-written digits from the
MNIST_ dataset.  Here are the first 8 images from MNIST:

.. image:: firsteightimages.jpg

The following loads the MNIST data:

.. _LeNet: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
.. _MNIST: http://yann.lecun.com/exdb/mnist

.. doctest::

    julia> include(Pkg.dir("Knet/examples/mnist.jl"))
    INFO: Loading MNIST...

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
28x28x1 array, where the first two numbers represent the width and
height in pixels, the third number is the number of channels (which is
1 for grayscale images, 3 for RGB images etc.)  The pixel values have
been normalized to :math:`[0,1]`.  Each output y is a ten-dimensional
one-hot vector (a vector that has a single non-zero component)
indicating the correct class (0-9) for a given image.

This is a much larger dataset than Housing.  For computational
efficiency, it is not advisable to use these examples one at a time
during training like we did before.  We will split the data into
groups of 100 examples called **minibatches**, and pass data to
``forw`` and ``back`` one minibatch at a time instead of one instance
at a time.  On a machine with a Nvidia K20 GPU, one epoch of training
LeNet on MNIST takes about 3.1 seconds with a minibatch size of 100,
10.8 seconds with a minibatch size of 10, and 75.2 seconds if we do
not use minibatches.  

Knet provides a small ``minibatch`` function to split the data:

.. testcode::

    function minibatch(x, y, batchsize)
        data = Any[]
        for i=1:batchsize:ccount(x)
            j=min(i+batchsize-1,ccount(x))
            push!(data, (cget(x,i:j), cget(y,i:j)))
        end
        return data
    end

.. testoutput:: :hide:

    ...

.. _iterables: http://julia.readthedocs.org/en/release-0.4/manual/interfaces/#iteration
.. _subarrays: http://julia.readthedocs.org/en/release-0.4/manual/arrays/

``minibatch`` takes ``batchsize`` columns of ``x`` and ``y`` at a
time, pairs them up and pushes them into a ``data`` array.  It works
for arrays of any dimensionality, treating the last dimension as
"columns".  This type of minibatching is fine for small datasets, but
it requires holding two copies of the data in memory.  For problems
with a large amount of data you may want to use subarrays_ or
iterables_.

Here is ``minibatch`` in action:

.. doctest::

    julia> batchsize=100;
    julia> trn = minibatch(MNIST.xtrn, MNIST.ytrn, batchsize)
    600-element Array{Any,1}:...
    julia> tst = minibatch(MNIST.xtst, MNIST.ytst, batchsize)
    100-element Array{Any,1}:...

Each element of ``trn`` and ``tst`` is an x, y pair that contains 100
examples::

    julia> trn[1]
    (28x28x1x100 Array{Float32,4}:
     ...,
     10x100 Array{Float32,2}:
     ...)

Here are some simple train and test functions that use this type of
minibatched data.  Note that they take the loss function as a third
argument:

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

Next, we compile the model and set the learning rate to 0.1, which
works well for this example.  We use two new :ref:`loss functions
<loss-table>`: ``softloss`` computes the cross entropy loss,
:math:`E(p\log\hat{p})`, commonly used for training classification
models and ``zeroone`` computes the zero-one loss which is the ratio
of predictions that were wrong.  I got 2.26% test error after one
epoch of training.  Your results may be slightly different on
different machines, or different runs on the same machine because of
non-determinism introduced by parallel GPU operations.

.. After one epoch of training I got 2.26% test error.  Your results may
.. be slightly different because some of the convolution operations are
.. non-deterministic.  You should be able to get the error down to 0.8%
.. in about 30 epochs of training.  You can compare this with some
.. benchmark results on the MNIST_ web page:

.. doctest::

   julia> net = compile(:lenet3);
   julia> setp(net; lr=0.1);
   julia> train(net, trn, softloss);
   julia> test(net, tst, zeroone)
   0.0226

Conditional Evaluation
----------------------

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

There are cases where you want to execute parts of a model
*conditionally*, e.g. only during training, or only during some parts
of the input in some sequence models.  Knet supports the use of
**runtime conditions** for this purpose.  We will illustrate the use
of conditions by implementing a training technique called dropout_ to
improve the generalization power of the LeNet model.

.. _dropout: http://jmlr.org/papers/v15/srivastava14a.html
.. _conditional evaluation: http://julia.readthedocs.org/en/release-0.4/manual/control-flow/#man-conditional-evaluation

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
set, but does not generalize equally well to the test set.

Dropout prevents overfitting by injecting random noise into the model.
Specifically, for each ``forw`` call during training, dropout layers
placed between two operations replace a random portion of their input
with zeros, and scale the rest to keep the total output the same.
During testing random noise would degrade performance, so we would
like to turn dropout off.  Here is one way to implement this in Knet::

    @knet function drop(x; pdrop=0, o...)
        if dropout
            return x .* rnd(init=Bernoulli(1-pdrop, 1/(1-pdrop)))
        else
            return x
        end
    end

The keyword argument ``pdrop`` specifies the probability of dropping an
input element.  The ``if ... else ... end`` block causes `conditional
evaluation`_ the way one would expect.  The variable ``dropout`` next to
``if`` is a global condition variable: it is not declared as an argument
to the function.  Instead, once a model with a ``drop`` operation is
compiled, the call to ``forw`` accepts ``dropout`` as an optional keyword
argument and passes it down as a global condition::

    forw(model, input; dropout=true)

This means every time we call ``forw``, we can change whether dropout
occurs or not.  During test time, we would like to stop dropout, so we
can call the model with ``dropout=false``::

    forw(model, input; dropout=false)

By default, all unspecified condition variables are false, so we could
also omit the condition during test time::

    forw(model, input)

Here is one way to add dropout to the LeNet model:

.. testcode::

    @knet function lenet4(x)
        a = conv_pool_layer(x; wdims=(5,5,1,20),  bdims=(1,1,20,1))
        b = conv_pool_layer(a; wdims=(5,5,20,50), bdims=(1,1,50,1))
        bdrop = drop(b; pdrop=0.5)
        c = relu_layer(bdrop; wdims=(500,800), bdims=(500,1))
        return softmax_layer(c; wdims=(10,500), bdims=(10,1))
    end

.. testoutput:: :hide:

    ...

Whenever the condition variable ``dropout`` is true, this will replace
half of the entries in the ``b`` array with zeros.  We need to pass
the condition to ``forw`` in our ``train`` function:

.. testcode::

    function train(f, data, loss)
        for (x,y) in data
            forw(f, x; dropout=true)
            back(f, y, loss)
            update!(f)
        end
    end

.. testoutput:: :hide:

    ...

During training, we will also reduce the learning rate whenever the
test error gets worse, another precaution against overfitting::

    net = compile(:lenet4)
    lrate = 0.1
    setp(net; lr=lrate)
    decay = 0.9
    lasterr = 1.0

    for epoch=1:100
        train(net, trn, softloss)
        trnerr = test(net, trn, zeroone)
        tsterr = test(net, tst, zeroone)
        println((epoch, lrate, trnerr, tsterr))
        if tsterr > lasterr
            lrate = decay*lrate
            setp(net; lr=lrate)
        end
        lasterr = tsterr
    end

In 100 epochs, this should converge to about 0.5% error, i.e. reduce
the total number of errors on the 10K test set from around 80 to
around 50.  Congratulations!  This is fairly close to the state of the
art compared to other benchmark results on the MNIST_ website::

    (1,0.1,0.020749999999999824,0.01960000000000001)
    (2,0.1,0.013699999999999895,0.01600000000000001)
    ...
    (99,0.0014780882941434613,0.0003333333333333334,0.005200000000000002)
    (100,0.0014780882941434613,0.0003666666666666668,0.005000000000000002)


Recurrent neural networks
-------------------------
.. read-before-write, simple rnn, lstm

.. _static variables: https://en.wikipedia.org/wiki/Static_variable

In this section we will see how to implement **recurrent neural
networks** (RNNs) in Knet.  All local variables in Knet functions are
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

Training with sequences
-----------------------
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


Some useful tables
------------------

.. _primitives-table:

   ===============================	==============================================================================
   Operator                		Description
   ===============================	==============================================================================
   :func:`par() <par>`			a parameter array, updated during training; kwargs: ``dims, init``
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

.. [#] `Broadcasting operations`_ are element-by-element binary
       operations on arrays of possibly different sizes, such as
       adding a vector to each column of a matrix.  They expand
       singleton dimensions in array arguments to match the
       corresponding dimension in the other array without using extra
       memory, and apply the given function elementwise.

.. [#] For detailed information about convolution and pooling, please
       see the documentation for CUDNN_ and `CUDNN.jl`_.

.. _compounds-table:

TODO: make a compounds table.

.. _rgen-table:

TODO: make an rgen table.

.. _loss-table:

TODO: make a loss fn table.

Knet functions to help train models: (TODO: list all functions covered
in tutorial)

================================= ==============================================================================
Function                	  Description
================================= ==============================================================================
:func:`forw(f,x) <forw>`	  returns the prediction of model ``f`` on input ``x``
:func:`back(f,y,lossfn) <back>`	  computes the loss gradients of ``f`` parameters based on the desired output ``y`` and a loss function ``lossfn``
:func:`update!(f) <update!>`	  updates the parameters of ``f`` using the gradients computed by ``back`` to reduce loss
================================= ==============================================================================

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


.. _colon character: http://julia.readthedocs.org/en/release-0.4/manual/metaprogramming#symbols
.. _Julia function definition: http://julia.readthedocs.org/en/release-0.4/manual/functions>
.. _Broadcasting operations: http://julia.readthedocs.org/en/release-0.4/manual/arrays/#broadcasting
.. _CUDNN: https://developer.nvidia.com/cudnn
.. _CUDNN.jl: https://github.com/JuliaGPU/CUDNN.jl

.. This looks a lot like a regular `Julia function definition`_ except
.. for the ``@knet`` macro.  However it is important to emphasize that
.. the ``@knet`` macro does not define ``lin`` as a regular Julia
.. function or variable.  Furthermore, only a restricted set of statement
.. types (e.g. assignment and return statements) and operators
.. (e.g. ``par``, ``*`` and ``.+``) can be used in a @knet function
.. definition.  A list of Knet primitive operators is given below:

.. .. Note that we need to escape Knet variable names using the `colon
.. .. character`_ just like we did for ``:lin`` when compiling.

.. ..
..    This defines ``f`` as an actual model (model or Net?) that we can
..    train and use for predictions (repeated).  Note that the colon
..    character preceding the name of our Knet function is required in the
..    compile expression.  (TODO: can we get rid of the colon with a macro?)
..    (TODO: The motivation behind this two step process, first defining a
..    Knet function then compiling it into a model, will become more clear
..    when we introduce compile time parameters.)


.. ..
..    Also note that ``lin`` is not defined as a regular Julia function or
..    variable.

..    .. doctest

..       julia> lin(5)
..       ERROR: UndefVarError: lin not defined

.. ..
..    So far it looks like all Knet gave us is a very complicated way to
..    define a very simple function.  So why would anybody bother defining a
..    @knet function with all the syntactic restrictions, limited number of
..    operators, need for compilation etc.?

.. There are many ways to reduce overfitting: more training data, a
.. smaller model with fewer parameters, regularization , and early
.. stopping can all help, and will be covered later (remember the
.. ``l1reg`` and ``l2reg`` from the :ref:`table of training options
.. <training-options-table>`).  For now let's focus on dropout.


TODO:

* add intro/conclusion at all levels.
* primitive ops
* colon and symbols
* broadcasting ops
* keyword arguments
* link Julia functions to Julia doc
* repeat, zero sizes and size inference, keyword args to compile(), rgen distributions.
* fix doctest again.
* find the table that shows tradeoff for minibatching.
* installation link is broken: http://www.sphinx-doc.org/en/stable/markup/inline.html
* size inference?
* introduce table of distributions, Bernoulli etc.
* rnn1: would be nice to use 0 for xsize at this point.  Also this is
  the second time we are using Xavier etc without much explanation.
* broadcasting, explain in minibatch?

