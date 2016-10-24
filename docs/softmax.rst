**********************
Softmax Classification
**********************

.. note::

   **Concepts:** classification, likelihood, softmax, one-hot vectors,
   zero-one loss, conditional likelihood, MLE, NLL, cross-entropy loss

We will introduce classification problems and some simple models for
classification.

Classification
--------------

Classification problems are supervised machine learning problems where
the task is to predict a discrete class for a given input (unlike
regression where the output was numeric).  A typical example is
handwritten digit recognition where the input is an image of a
handwritten digit, and the output is one of the discrete categories
:math:`\{0, \ldots, 9\}`.  As in all supervised learning problems the
training data consists of a set of example input-output pairs.

Likelihood
----------

A natural objective in classification could be to minimize the number
of misclassified examples in the training data.  This number is known
as the **zero-one loss**.  However the zero-one loss has some
undesirable properties for training: in particular it is
discontinuous.  A small change in one of the parameters either has no
effect on the loss, or can turn one or more of the predictions from
false to true or true to false, causing a discontinuous jump in the
objective.  This means the gradient of the zero-one loss with respect
to the parameters is either undefined or not helpful.

A more commonly used objective for classification is conditional
likelihood: the probability of the observed data given our model *and
the inputs*.  Instead of predicting a single class for each instance,
we let our model predict a probability distribution over all classes.
Then we adjust the weights of the model to increase the probabilities
for the correct classes and decrease it for others.  This is also
known as the **maximum likelihood estimation** (MLE).

Let :math:`\mathcal{X}=\{x_1,\ldots,x_N\}` be the inputs in the
training data, :math:`\mathcal{Y}=\{y_1,\ldots,y_N\}` be the correct
classes and :math:`\theta` be the parameters of our model.
Conditional likelihood is:

.. math::

   L(\theta) = P(\mathcal{Y}|\mathcal{X},\theta) 
   = \prod_{n=1}^N P(y_n|x_n,\theta)

The second equation assumes that the data instances were generated
independently.  We usually work with log likelihood for mathematical
convenience: log is a monotonically increasing function, so maximizing
likelihood is the same as maximizing log likelihood:

.. math::

   \ell(\theta) = \log P(\mathcal{Y}|\mathcal{X},\theta) 
   = \sum_{n=1}^N \log P(y_n|x_n,\theta)

We will typically use the negative of :math:`\ell` (machine learning
people like to minimize), which is known as **negative log
likelihood** (NLL), or **cross-entropy loss**.

Softmax
-------

The linear regression model we have seen earlier produces unbounded
:math:`y` values.  To go from arbitrary values
:math:`y\in\mathbb{R}^C` to normalized probability estimates
:math:`p\in\mathbb{R}^C` for a single instance, we use exponentiation
and normalization:

.. math::

   p_i = \frac{\exp y_i}{\sum_{c=1}^C \exp y_c}

where :math:`i,c\in\{1,\ldots,C\}` range over classes, and :math:`p_i,
y_i, y_c` refer to class probabilities and values for a single
instance.  This is called the **softmax function**.  A model that
converts the unnormalized values at the end of a linear regression to
normalized probabilities for classification is called the **softmax
classifier**.

We need to figure out the backward pass for the softmax function.  In
other words if someone gives us the gradient of some objective
:math:`J` with respect to the class probabilities :math:`p` for a
single training instance, what is the gradient with respect to the
input of the softmax :math:`y`?  First we'll find the partial
derivative of one component of :math:`p` with respect to one component
of :math:`y`:

.. math::

   \frac{\partial p_i}{\partial y_j}
   &=& \frac{[i=j] \exp y_i \sum_c \exp y_c - \exp y_i \exp y_j}
            {(\sum_c \exp y_c)^2} \\
   &=& \,[i=j]\, p_i - p_i p_j

The square brackets are the `Iverson bracket`_ notation,
i.e. :math:`[A]` is 1 if :math:`A` is true, and 0 if :math:`A` is
false.  

.. _Iverson bracket: https://en.wikipedia.org/wiki/Iverson_bracket

Note that a single entry in :math:`y` effects :math:`J` through
multiple paths (:math:`y_j` contributes to the denominator of every
:math:`p_i`), and these effects need to be added for :math:`\partial
J/\partial y_j`:

.. math::

   \frac{\partial J}{\partial y_j}
   = \sum_{i=1}^C \frac{\partial J}{\partial p_i}
   \frac{\partial p_i}{\partial y_j}


One-hot vectors
---------------

When using a probabilistic classifier, it is convenient to represent
the desired output as a **one-hot vector**, i.e. a vector in which all
entries are '0' except a single '1'.  If the correct class is
:math:`c\in\{1,\ldots,C\}`, we represent this with a one-hot vector
:math:`p\in\mathbb{R}^C` where :math:`p_c = 1` and :math:`p_{i\neq c}
= 0`.  Note that :math:`p` can be viewed as a probability vector where
all the probability mass is concentrated at `c`.  This representation
also allows us to have probabilistic targets where there is not a
single answer but target probabilities associated with each answer.
Given a one-hot (or probabilistic) :math:`p`, and the model prediction
:math:`\hat{p}`, we can write the log-likelihood for a single instance
as:

.. math::

   \ell = \sum_{c=1}^C p_c \log \hat{p}_c


Gradient of log likelihood
--------------------------

To compute the gradient for log likelihood, we need to make the
normalization of :math:`\hat{p}` explicit:

.. TODO: explain why we need explicit normalization.

.. math::

   \ell &=& \sum_c p_c \log \frac{\hat{p}_c}{\sum_k\hat{p}_k} \\
   &=& \sum_c p_c \log{\hat{p}_c} - \sum_c p_c \log \sum_k\hat{p}_k \\
   &=& (\sum_c p_c \log{\hat{p}_c}) - (\log \sum_k\hat{p}_k) \\
   \frac{\partial \ell}{\partial \hat{p}_i} &=&
   \frac{p_i}{\hat{p}_i} - \frac{1}{\sum_k\hat{p}_k}
   = \frac{p_i}{\hat{p}_i} - 1

The gradient with respect to unnormalized y takes a particularly
simple form:

.. math::

   \frac{\partial\ell}{\partial y_j}
   &=& \sum_i \frac{\partial\ell}{\partial \hat{p}_i}
   \frac{\partial \hat{p}_i}{\partial y_j} \\
   &=& \sum_i (\frac{p_i}{\hat{p}_i} - 1)(\,[i=j]\, \hat{p}_i - \hat{p}_i \hat{p}_j) \\
   &=& \, p_j - \hat{p}_j \\
   \nabla\ell &=& \, p - \hat{p}

The gradient with respect to :math:`\hat{p}` causes numerical overflow
when some components of :math:`\hat{p}` get very small.  In practice
we usually skip that and directly compute the gradient with respect to
:math:`y` which is numerically stable.

MNIST example
-------------

.. _MNIST: http://yann.lecun.com/exdb/mnist

Let's try our softmax classifier on the MNIST_ handwritten digit
classification dataset.  Here are the first 8 images from MNIST, the goal is
to look at the pixels and classify each image as one of the digits
0-9:

.. image:: images/firsteightimages.jpg

See :ref:`training-with-minibatches` for more information about the
MNIST task, loading and minibatching data, and simple train and test
scripts.

Here is the softmax classifier in Julia:

.. code::

   function softmax(w,x,ygold)
       ypred = w[1] * x .+ w[2]
       return softloss(ygold, ypred)
   end

   function softloss(ygold, ypred)
       ynorm = ypred .- log(sum(exp(ypred),1))
       -sum(ygold .* ynorm) / size(ygold,2)
   end

   softmax_gradient = grad(softmax)

.. code:: :hide:

   ...

Let us train our model for 100 epochs and print out the classification
error on the training and test sets after every epoch (see the full
example in Pkg.dir("Knet/examples/mnist.jl")):

.. code::

    w = Any[0.1*randn(10,784), zeros(10,1)]
    for epoch=1:nepochs
        for (x,y) in dtrn  # dtrn is a list of minibatches
            g = softmax_gradient(w, x, y)
            for i in 1:length(w)
                w[i] -= lr * g[i]
            end
        end
	# Print accuracy
    end

.. testoutput::

   (:epoch,0,:trn,0.1135,:tst,0.1097)
   (:epoch,1,:trn,0.9008666666666667,:tst,0.9048)
   ...
   (:epoch,99,:trn,0.9274833333333333,:tst,0.9177)
   (:epoch,100,:trn,0.92755,:tst,0.9176)

Here is a plot of the losses vs training epochs:

.. image:: images/mnist_softmax.png

We can observe a few things.  First the training losses are better
than the test losses.  This means there is some **overfitting**.
Second, it does not look like the training loss is going down to zero.
This means the softmax model is not flexible enough to fit the
training data exactly.

Representational power
----------------------

So far we have seen how to create a machine learning model as a
differentiable program (linear regression, softmax classification)
whose parameters can be adjusted to hopefully imitate whatever process
generated our training data.  A natural question to ask is whether a
particular model can behave like any system we want (given the right
parameters) or whether there is a limit to what it can represent.


It turns out the softmax classifier is quite limited in its
representational power: it can only represent linear classification
boundaries.  To show this, remember the form of the softmax classifier
which gives the probability of the i'th class as:

.. math::

   p_i &=& \frac{\exp y_i}{\sum_{c=1}^C \exp y_c} \\

where :math:`y_i` is a linear function of the input :math:`x`.  Note
that :math:`p_i` is a monotonically increasing function of
:math:`y_i`, so for two classes :math:`i` and :math:`j`, :math:`p_i >
p_j` if :math:`y_i > y_j`.  The boundary between two classes :math:`i`
and :math:`j` is the set of inputs for which the probability of the
two classes are equal:

.. math::

   p_i &=& p_j \\
   y_i &=& y_j \\
   w_i x + b_i &=& w_j x + b_j \\
   (w_i - w_j) x + (b_i - b_j) &=& 0

where :math:`w_i, b_i` refer to the i'th row of :math:`w` and
:math:`b`. This is a linear equation, i.e. the border between two
classes will always be linear in the input space with the softmax
classifier:

.. image:: images/linear-boundary.png

In the MNIST example, the relation between the pixels and the digit
classes is unlikely to be this simple.  That is why we are stuck at
6-7% training error.  To get better results we need more powerful
models.

References
----------

* http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression
