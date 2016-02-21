**********************
Multilayer Perceptrons
**********************

In this section we create multilayer perceptrons by stacking multiple
linear layers with non-linear activation functions in between.

Stacking linear classifiers is useless
--------------------------------------

We could try stacking multiple linear classifiers together:

.. testcode::

    @knet function mnist_softmax_2(x)
        w1 = par(init=Gaussian(0,0.001), dims=(100,28*28))
        b1 = par(init=Constant(0), dims=(100,1))
        y1 = w1 * x + b1
        w2 = par(init=Gaussian(0,0.001), dims=(10,100))
        b2 = par(init=Constant(0), dims=(10,1))
	return soft(w2 * y1 + b2)
    end

.. testoutput:: :hide:

   ...

Note that instead of outputting the softmax of ``y1``, we used it as
input to another softmax classifier.  Such intermediate arrays are
known as **hidden layers** because their outputs are not directly
visible outside the model.

If you experiment with this model (I suggest using a smaller learning
rate, e.g. 0.01), you will see that it performs similarly to the
original softmax model.  The reason is simple:

.. math::

   \hat{p} &=& \mbox{soft}(W_2 (W_1 x + b_1) + b_2) \\
   &=& \mbox{soft} ((W_2 W_1)\, x + W_2 b_1 + b_2)

In other words, we still have a linear classifier!  No matter how many
linear functions you put on top of each other, what you get at the end
is still a linear function.  So this model has exactly the same
representation power as the softmax model.  Unless, we add a simple
instruction...

Introducing nonlinearities
--------------------------

Here is a slightly modified version of the two layer softmax called
the **multilayer perceptron** (MLP):

.. testcode::

    @knet function mnist_mlp(x)
        w1 = par(init=Gaussian(0,0.001), dims=(100,28*28))
        b1 = par(init=Constant(0), dims=(100,1))
        y1 = relu(w1 * x + b1)
        w2 = par(init=Gaussian(0,0.001), dims=(10,100))
        b2 = par(init=Constant(0), dims=(10,1))
	return soft(w2 * y1 + b2)
    end

.. testoutput:: :hide:

   ...

The only difference is the ``relu`` function we introduced in line 4.
This is known as the rectified linear unit (or rectifier), and is a
simple function defined by ``relu(x)=max(x,0)`` applied elementwise to
the input array.  But what a difference it makes to the model!  Here
are the learning curves for ``mnist_mlp``:

.. image:: images/mnist_mlp.png

Here are the learning curves for ``softmax`` plotted at the same scale
for comparison:

.. image:: images/mnist_softmax2.png

We can observe a few things: using MLP instead of a linear model
brings the training error from 6.7% to 0 and the test error from 7.5%
to 2.0%.  There is still overfitting: the test error is not as good as
the training error, but the model has no problem classifying the training
data (all 60,000 examples) perfectly.

Types of nonlinearities (activation functions)
----------------------------------------------

The functions we throw between linear layers to break the linearity
are called **nonlinearities** or **activation functions**.  Here are
some activation functions that have been used as nonlinearities:

.. image:: images/actf.png

The step functions were the earliest activation functions used in the
perceptrons of 1950s.  Unfortunately they do not give a useful
derivative that can be used for training a multilayer model.  Sigmoid
and tanh (``sigm`` and ``tanh`` in Knet) became popular in 1980s as
smooth approximations to the step functions and allowed the
application of the backpropagation algorithm.  Modern activation
functions like relu and maxout are piecewise linear.  They are
computationally inexpensive (no exponentials), and perform well in
practice.  We are going to use relu in most of our models.  Here is
the backward passes for sigmoid, tanh, and relu:

======== ========================================= ========
function forward                                   backward
======== ========================================= ========
sigmoid  :math:`y = \frac{1}{1+e^{-x}}`            :math:`\nabla_x J = y\,(1-y) \nabla_y J`
tanh     :math:`y = \frac{e^x-e^{-x}}{e^x+e^{-x}}` :math:`\nabla_x J = (1+y)(1-y) \nabla_y J`
relu     :math:`y = \max(0,x)`                     :math:`\nabla_x J = [ y \geq 0 ] \nabla_y J`
======== ========================================= ========


Representational power
----------------------

You might be wondering whether relu had any special properties or
would any of the other nonlinearities be sufficient.  Another question
is whether there are functions multilayer perceptrons cannot represent
and if so whether adding more layers or different types of functions
would increase their representational power.  The short answer is that
a multilayer perceptron with a single large enough hidden layer can
approximate any function, and can do so with any of the nonlinearities
introduced in the last section.  Multilayer perceptrons are universal
function approximators!

Please see `(Nielsen, 2016, Ch 4)`_ for an intuitive explanation of
this result and `(Bengio et al. 2016, Ch 6.4)`_ for a more in depth
discussion and references.

.. _(Nielsen, 2016, Ch 4): http://neuralnetworksanddeeplearning.com/chap4.html

.. _(Bengio et al. 2016, Ch 6.4): http://www.deeplearningbook.org/contents/mlp.html

.. universality: nielsen constructs it turning step activations into
.. bump functions to approx a given function.  He uses two hidden
.. layers but argues one is enough.  

.. I thought another argument was to restrict the test to a finite
.. number of input points, and just get the right answers for the
.. training data, each hidden unit representing one training sample.

.. nand gates can compute any boolean function.

.. why it is not enough, boolean argument? neither nielsen nor
.. karpathy makes the boolean argument showing two layer net requires
.. exponentially more units than three layer for some functions.

.. neuron picture: needed for the nielsen argument

.. what else? check karpathy. talks about overfitting, has some good
.. arguments for not using network size to prevent overfitting: large
.. networks may have many more local minima but they have similar
.. performance, vs small networks have few bad local minima making
.. optimization more difficult.  so it is better to use dropout etc.

.. http://www.deeplearningbook.org/contents/mlp.html 6.4:
.. representation vs learnability.  talks about sets of functions that
.. require exponentially more units for shallow networks.  number of
.. bool fns with n inputs is 2^2^n, so we'll need 2^n bits of info in
.. the net to distinguish.  one hidden unit per training example
.. argument.  points to some recent proofs involving relu and abs
.. units that discuss representational efficiency.


.. TODO: the neural net vs matrix pictures.
