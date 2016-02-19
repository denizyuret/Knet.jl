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
likelihood** (NLL), or **cross-entropy loss** (``softloss`` in Knet).

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
y_i, y_c` refer to class probabilities and values for a single instance.
This is called the **softmax function** (``soft`` operation in Knet).
A model that converts the unnormalized values at the end of a linear
regression to normalized probabilities for classification is called
the **softmax classifier**.

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

.. softmax classifier

