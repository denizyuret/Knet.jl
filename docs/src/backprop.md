***************
Backpropagation
***************

.. note::

   **Concepts:** supervised learning, training data, regression,
   squared error, linear regression, stochastic gradient descent

.. TODO: rewrite this whole introduction
.. TODO: add explanation for dw = dy * x^T - everybody gets confused
.. TODO: integrate knet examples
.. TODO: integrate quiz questions as exercises
.. TODO: add programming examples/exercises from class logs

Arthur Samuel, the author of the first self-learning checkers program,
defined machine learning as a "field of study that gives computers the
ability to learn without being explicitly programmed".  This leaves
the definition of learning a bit circular.  Tom M. Mitchell provided a
more formal definition: "A computer program is said to learn from
experience E with respect to some class of tasks T and performance
measure P if its performance at tasks in T, as measured by P, improves
with experience E," where the task, the experience, and the
performance measure are to be specified based on the problem.

We will start with **supervised learning**, where the task is to
predict the output of an unknown system given its input, and the
experience consists of a set of example input-output pairs, also known
as the **training data**.  When the outputs are numeric such problems
are called **regression**.  In **linear regression** we use a linear
function as our model:

.. math::

   \hat{y} = W x + b

Here :math:`x` is the model input, :math:`\hat{y}` is the model
output, :math:`W` is a matrix of weights, and :math:`b` is a vector of
biases.  By adjusting the parameters of this model, i.e. the weights
and the biases, we can make it compute any linear function of
:math:`x`.

"All models are wrong, but some models are useful." George Box
famously said.  We do not necessarily know that the system whose
output we are trying to predict is governed by a linear
relationship. All we know is a finite number of input-output examples:

.. math::

   \mathcal{D}=\{(x_1,y_1),\ldots,(x_N,y_N)\}

It is just that we have to start model building somewhere and the set
of all linear functions is a good place to start for now.

.. TODO: also mention somewhere the idea of basis functions

.. decide on a consistent set of index letters: inputs, outputs, instances
.. murphy uses N for instances, D for x dimensions
.. bishop uses D for x dims, M for phi dims, n=1:N for instances, K
.. for y dims for classification, (regression assumes 1 dim). we could
.. also use C, so the weight matrix is CxD.

A commonly used performance measure in regression problems is the
**squared error**, i.e. the average squared difference between the
actual output values and the ones predicted by the model.  So our goal
is to find model parameters that minimize the squared error:

.. math::

   \arg\min_{W,b} \frac{1}{N} \sum_{n=1}^N \| \hat{y}_n - y_n \|^2

Where :math:`\hat{y}_n = W x_n + b` denotes the output predicted by
the model for the :math:`n` th example.  

There are several methods to find the solution to the problem of
minimizing squared error.  Here we will present the **stochastic
gradient descent** (SGD) method because it generalizes well to more
complex models.  In SGD, we take the training examples one at a time
(or in small groups called minibatches), compute the gradient of the
error with respect to the parameters, and move the parameters a small
step in the direction that will decrease the error.  First some notes
on the math.

Partial derivatives
-------------------

When we have a function with several inputs and one output, we can
look at how the function value changes in response to a small change
in one of its inputs holding the rest fixed.  This is called a partial
derivative.  Let us consider the squared error for the :math:`n` th
input as an example:

.. math::

   J = \| W x_n + b - y_n \|^2

So the partial derivative :math:`\partial J / \partial w_{ij}` would
tell us how many units :math:`J` would move if we moved :math:`w_{ij}`
in :math:`W` one unit (at least for small enough units).  Here is a
more graphical representation:

.. image:: images/linregforw.jpg

In this figure, it is easier to see that the machinery that generates
:math:`J` has many "inputs".  In particular we can talk about how
:math:`J` is effected by changing parameters :math:`W` and :math:`b`,
as well as changing the input :math:`x`, the model output
:math:`\hat{y}`, the desired output :math:`y`, or intermediate values
like :math:`z` or :math:`r`.  So partial derivatives like
:math:`\partial J / \partial x_i` or :math:`\partial J / \partial
\hat{y}_j` are fair game and tell us how :math:`J` would react in
response to small changes in those quantities.

Chain rule
----------

The chain rule allows us to calculate partial derivatives in terms of
other partial derivatives, simplifying the overall computation.  We
will go over it in some detail as it forms the basis of the
backpropagation algorithm.  For now let us assume that each of the
variables in the above example are scalars.  We will start by looking
at the effect of :math:`r` on :math:`J` and move backward from there.
Basic calculus tells us that:

.. math::

   J = r^2 \\
   {\partial J}/{\partial r} = 2r

Thus, if :math:`r=5` and we decrease :math:`r` by a small
:math:`\epsilon`, the squared error :math:`J` will go down by
:math:`10\epsilon`.  Now let's move back a step and look at
:math:`\hat{y}`:

.. math::

   r = \hat{y} - y \\
   {\partial r}/{\partial \hat{y}} = 1

So how much effect will a small :math:`\epsilon` decrease in
:math:`\hat{y}` have on :math:`J` when :math:`r=5`?  Well, when
:math:`\hat{y}` goes down by :math:`\epsilon`, so will :math:`r`,
which means :math:`J` will go down by :math:`10\epsilon` again.  The
chain rule expresses this idea:

.. math::

   \frac{\partial J}{\partial\hat{y}} = 
   \frac{\partial J}{\partial r}
   \frac{\partial r}{\partial\hat{y}}
   = 2r

Going back further, we have:

.. math::

   \hat{y} = z + b \\
   {\partial \hat{y}}/{\partial b} = 1 \\
   {\partial \hat{y}}/{\partial z} = 1 \\

Which means :math:`b` and :math:`z` have the same effect on :math:`J`
as :math:`\hat{y}` and :math:`r`, i.e. decreasing them by
:math:`\epsilon` will decrease :math:`J` by :math:`2r\epsilon` as
well.  Finally:

.. math::

   z = w x \\
   {\partial z}/{\partial x} = w \\
   {\partial z}/{\partial w} = x

This allows us to compute the effect of :math:`w` on :math:`J` in
several steps: moving :math:`w` by :math:`\epsilon` will move
:math:`z` by :math:`x\epsilon`, :math:`\hat{y}` and :math:`r` will
move exactly the same amount because their partials with :math:`z` are
1, and finally since :math:`r` moves by :math:`x\epsilon`, :math:`J`
will move by :math:`2rx\epsilon`.

.. math::

   \frac{\partial J}{\partial w} =
   \frac{\partial J}{\partial r}
   \frac{\partial r}{\partial \hat{y}}
   \frac{\partial \hat{y}}{\partial z}
   \frac{\partial z}{\partial w}
   = 2rx

We can represent this process of computing partial derivatives as
follows:

.. image:: images/linregback.jpg

Note that we have the same number of boxes and operations, but all the
arrows are reversed.  Let us call this the backward pass, and the
original computation in the previous picture the forward pass.  Each
box in this backward-pass picture represents the partial derivative
for the corresponding box in the previous forward-pass picture.  Most
importantly, each computation is local: each operation takes the
partial derivative of its output, and multiplies it with a factor that
only depends on the original input/output values to compute the
partial derivative of its input(s).  In fact we can implement the
forward and backward passes for the linear regression model using the
following local operations:

.. image:: images/sqnorm.jpg

.. image:: images/sub.jpg

.. image:: images/add.jpg

.. image:: images/mul.jpg


Multiple dimensions
-------------------

Let's look at the case where the input and output are not scalars but
vectors.  In particular assume that :math:`x \in \mathbb{R}^D` and
:math:`y \in \mathbb{R}^C`.  This makes :math:`W \in
\mathbb{R}^{C\times D}` a matrix and :math:`z,b,\hat{y},r` vectors in
:math:`\mathbb{R}^C`.  During the forward pass, :math:`z=Wx` operation
is now a matrix-vector product, the additions and subtractions are
elementwise operations.  The squared error :math:`J=\|r\|^2=\sum
r_i^2` is still a scalar.  For the backward pass we ask how much each
element of these vectors or matrices effect :math:`J`.  Starting with
:math:`r`:

.. math::

   J = \sum r_i^2 \\
   {\partial J}/{\partial r_i} = 2r_i

We see that when :math:`r` is a vector, the partial derivative of each
component is equal to twice that component.  If we put these partial
derivatives together in a vector, we obtain a **gradient** vector:

.. math::

   \nabla_r J
   \equiv \langle \frac{\partial J}{\partial r_1}, \cdots, \frac{\partial J}{\partial r_C} \rangle
   = \langle 2 r_1, \ldots, 2 r_C \rangle 
   = 2\vec{r}

The addition, subtraction, and square norm operations work the same
way as before except they act on each element.  Moving back through
the elementwise operations we see that:

.. math::

   \nabla_r J = \nabla_\hat{y} J = \nabla_b J = \nabla_z J = 2\vec{r}

For the operation :math:`z=Wx`, a little algebra will show you that:

.. math::

   \nabla_W J = \nabla_z J \cdot x^T \\
   \nabla_x J = W^T \cdot \nabla_z J

Note that the gradient of a variable has the same shape as the
variable itself.  In particular :math:`\nabla_W J` is a :math:`C\times
D` matrix.  Here is the graphical representation for matrix
multiplication:

.. image:: images/dot.jpg


Multiple instances
------------------

We will typically process data multiple instances at a time for
efficiency.  Thus, the input :math:`x` will be a :math:`D\times N`
matrix, and the output :math:`y` will be a :math:`C\times N` matrix,
the :math:`N` columns representing :math:`N` different instances.
Please verify to yourself that the forward and backward operations as
described above handle this case without much change: the elementwise
operations act on the elements of the matrices just like vectors, and
the matrix multiplication and its gradient remains the same.  Here is
a picture of the forward and backward passes:

.. image:: images/batchforwback.jpg

The only complication is at the addition of the bias vector.  In the
batch setting, we are adding :math:`b\in\mathbb{R}^{C\times 1}` to
:math:`z\in\mathbb{R}^{C\times N}`.  This will be a broadcasting
operation, i.e. the vector :math:`b` will be added to each column of
the matrix :math:`z` to get :math:`\hat{y}`.  In the backward pass,
we'll need to add the columns of :math:`\nabla_\hat{y} J` to get the
gradient :math:`\nabla_b J`.

.. TODO: is the Jacobian relevant here?


Stochastic Gradient Descent
---------------------------

The gradients calculated by backprop, :math:`\nabla_w J` and
:math:`\nabla_b J`, tell us how much small changes in corresponding
entries in :math:`w` and :math:`b` will effect the error (for the last
instance, or minibatch). Small steps in the gradient direction will
increase the error, steps in the opposite direction will decrease the
error.

In fact, we can show that the gradient is the direction of steepest
ascent.  Consider a unit vector :math:`v` pointing in some arbitrary
direction.  The rate of change in this direction is given by the
projection of :math:`v` onto the gradient, i.e. their dot product
:math:`\nabla J \cdot v`.  What direction maximizes this dot product?
Recall that:

.. math::

   \nabla J \cdot v = | \nabla J |\,\, | v | \cos(\theta)

where :math:`\theta` is the angle between :math:`v` and the gradient
vector.  :math:`\cos(\theta)` is maximized when the two vectors point
in the same direction.  So if you are going to move a fixed (small)
size step, the gradient direction gives you the biggest bang for the
buck.

.. This is from:
.. http://math.stackexchange.com/questions/223252/why-is-gradient-the-direction-of-steepest-ascent
.. Another longer explanation:
.. http://betterexplained.com/articles/understanding-pythagorean-distance-and-the-gradient

This suggests the following update rule:

.. math::

   w \leftarrow w - \nabla_w J

This is the basic idea behind Stochastic Gradient Descent (SGD): Go
over the training set instance by instance (or minibatch by
minibatch). Run the backpropagation algorithm to calculate the error
gradients. Update the weights and biases in the opposite direction of
these gradients. Rinse and repeat...

Over the years, people have noted many subtle problems with this
approach and suggested improvements:

**Step size:** If the step sizes are too small, the SGD algorithm will
take too long to converge. If they are too big it will overshoot the
optimum and start to oscillate. So we scale the gradients with an
adjustable parameter called the learning rate :math:`\eta`:

.. math::

   w \leftarrow w - \eta \nabla_w J

**Step direction:** More importantly, it turns out the gradient (or
its opposite) is often NOT the direction you want to go in order to
minimize error. Let us illustrate with a simple picture:

.. image:: images/longnarrowvalley.png

.. from: http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_Martens10.pdf

The figure on the left shows what would happen if you stood on one
side of the long narrow valley and took the direction of steepest
descent: this would point to the other side of the valley and you
would end up moving back and forth between the two sides, instead of
taking the gentle incline down as in the figure on the right.  The
direction across the valley has a high gradient but also a high
curvature (second derivative) which means the descent will be sharp
but short lived.  On the other hand the direction following the bottom
of the valley has a smaller gradient and low curvature, the descent
will be slow but it will continue for a longer distance.  `Newton's
method`_ adjusts the direction taking into account the second
derivative:

.. _Newton's method: https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization

.. image:: images/330px-Newton_optimization_vs_grad_descent.svg.png

In this figure, the two axes are w1 and w2, two parameters of our
network, and the contour plot represents the error with a minimum at
x. If we start at x0, the Newton direction (in red) points almost
towards the minimum, whereas the gradient (in green), perpendicular to
the contours, points to the right.

Unfortunately Newton's direction is expensive to compute. However, it
is also probably unnecessary for several reasons: (1) Newton gives us
the ideal direction for second degree objective functions, which our
objective function almost certainly is not, (2) The error function
whose gradient backprop calculated is the error for the last
minibatch/instance only, which at best is a very noisy approximation
of the real error function, thus we shouldn't spend too much effort
trying to get the direction exactly right.  

So people have come up with various approximate methods to improve the
step direction. Instead of multiplying each component of the gradient
with the same learning rate, these methods scale them separately using
their running average (momentum, Nesterov), or RMS (Adagrad, Rmsprop).
Some even cap the gradients at an arbitrary upper limit (gradient
clipping) to prevent unstabilities.

You may wonder whether these methods still give us directions that
consistently increase/decrease the objective function.  If we do not
insist on the *maximum* increase, any direction whose components have
the same signs as the gradient vector is guaranteed to increase the
function (for short enough steps).  The reason is again given by the
dot product :math:`\nabla J \cdot v`.  As long as these two vectors
carry the same signs in the same components, the dot product, i.e. the
rate of change along :math:`v`, is guaranteed to be positive.

**Minimize what?** The final problem with gradient descent, other than
not telling us the ideal step size or direction, is that it is not
even minimizing the right objective! We want small error on never
before seen test data, not just on the training data. The truth is, a
sufficiently large model with a good optimization algorithm can get
arbitrarily low error on any finite training data (e.g. by just
memorizing the answers). And it can typically do so in many different
ways (typically many different local minima for training error in
weight space exist). Some of those ways will generalize well to unseen
data, some won't. And unseen data is (by definition) not seen, so how
will we ever know which weight settings will do well on it? 

There are at least three ways people deal with this problem: (1) Bayes
tells us that we should use all possible models and weigh their
answers by how well they do on training data (see Radford Neal's fbm),
(2) New methods like dropout that add distortions and noise to inputs,
activations, or weights during training seem to help generalization,
(3) Pressuring the optimization to stay in one corner of the weight
space (e.g. L1, L2, maxnorm regularization) helps generalization.

References
----------

* http://ufldl.stanford.edu/tutorial/supervised/LinearRegression

.. The gradient of a function with respect to a vector or matrix is not
.. only a collection of partial derivatives, but also the *direction of
.. maximum ascent*, i.e. if you are going to take a fixed size (small)
.. step, and you want the maximum increase in the function, you should
.. step in the direction of the gradient.  In our case, we want the
.. maximum *decrease* in the squared error function, so we should step in
.. the opposite direction of the gradient:

.. .. math::

..    w \leftarrow w - \eta \nabla_w J

.. where :math:`w` is a parameter we want to optimize and :math:`\eta` is
.. a scalar that controls the step-size known as the learning rate.
.. Before we close this section we will consider several questions that
.. will be helpful when understanding more advanced optimization
.. techniques:

.. 1. Why is the gradient the direction of maximum ascent?  Take a look at
.. the gradient again:

.. .. math::

..    \nabla_r J
..    \equiv \langle \frac{\partial J}{\partial r_1}, \cdots, \frac{\partial J}{\partial r_C} \rangle

.. The components of the gradient vector are simply the rates of change in
.. the objective function if we move in one dimension, holding all others
.. fixed.  So why don't we simply move along the dimension that would
.. give us the highest rate of change?  

.. Consider a unit vector :math:`v` pointing in some arbitrary direction.
.. The rate of change in this direction is given by the projection of the
.. gradient onto :math:`v`, i.e. their dot product :math:`\nabla J \cdot
.. v`.  What direction maximizes this dot product?  Recall that:

.. .. math::

..    \nabla J \cdot v = | \nabla J |\,\, | v | \cos(\theta)

.. where :math:`\theta` is the angle between :math:`v` and the gradient
.. vector.  :math:`\cos(\theta)` is maximized when the two vectors point
.. in the same direction.  That is why the gradient is the direction of
.. maximum ascent.

.. .. This is from:
.. .. http://math.stackexchange.com/questions/223252/why-is-gradient-the-direction-of-steepest-ascent
.. .. Another longer explanation:
.. .. http://betterexplained.com/articles/understanding-pythagorean-distance-and-the-gradient

.. 2. Are there other directions that are guaranteed to increase or
.. decrease the objective function?  


.. 3. Are there times when we want to move in directions other than the
.. gradient?  Yes, and there are lots of potential reasons:

.. * The gradients we compute are based on a finite sample of instances
..   (sometimes just a single instance, or a small minibatch).  As such,
..   they are only noisy approximations to the true gradient.  This does
..   not necessarily tell us what directions may be better, but warns us
..   not to trust the gradient we have at some point too much.

.. * The local direction of maximum descent does not necessarily point to
..   the global minimum, as this figure illustrates:

.. .. image:: images/4_Minimize.anc7.gif

