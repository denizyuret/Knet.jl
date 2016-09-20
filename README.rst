Introduction to Knet
====================

.. image:: https://travis-ci.org/denizyuret/Knet.jl.svg?branch=master
   :target: https://travis-ci.org/denizyuret/Knet.jl

.. image:: http://pkg.julialang.org/badges/Knet_0.4.svg
   :target: http://pkg.julialang.org/?pkg=Knet

.. image:: http://pkg.julialang.org/badges/Knet_0.5.svg
   :target: http://pkg.julialang.org/?pkg=Knet
	    

`Knet <http://knet.rtfd.org>`__ (pronounced "kay-net") is the `Koç
University <http://www.ku.edu.tr/en>`__ deep learning framework
implemented in `Julia <http://julia.rtfd.org>`__ by `Deniz Yuret
<http://www.denizyuret.com>`__ and collaborators. It supports
construction of high-performance deep learning models in plain Julia
by combining automatic differentiation with efficient GPU kernels and
memory management. Models can be defined and trained using arbitrary
Julia code with helper functions, loops, conditionals, recursion,
closures, array indexing and concatenation. The training can be
performed on the GPU by simply using KnetArray instead of Array for
parameters and data. Check out the `full documentation
<http://knet.rtfd.org>`__ and the `examples directory
<https://github.com/denizyuret/Knet.jl/tree/master/examples>`__ for
more information.

Contents
--------

-  `Installation`_
-  `Examples`_

   -  `Linear regression`_
   -  `Softmax classification`_
   -  `Multi-layer perceptron`_
   -  `Convolutional neural network`_
   -  `Recurrent neural network`_

-  `Under the hood`_
-  `Benchmarks`_
-  `Contributing`_

Installation
------------

You can install Knet using ``Pkg.add("Knet")``. Some of the examples use
additional packages such as ArgParse, GZip, and JLD. These are not
required by Knet and can be installed when needed using additional
``Pkg.add()`` commands. See the detailed
`installation
instructions <http://knet.readthedocs.org/en/latest/install.html#installation>`__
as well as the section on `using Amazon
AWS <http://knet.readthedocs.org/en/latest/install.html#using-amazon-aws>`__
to experiment with GPU machines on the cloud with pre-installed Knet
images.

Examples
--------

In Knet, a machine learning model is defined using plain Julia code. A
typical model consists of a *prediction* and a *loss* function. The
prediction function takes model parameters and some input, returns the
prediction of the model for that input. The loss function measures how
bad the prediction is with respect to some desired output. We train a
model by adjusting its parameters to reduce the loss. In this section we
will see the prediction, loss, and training functions for five models:
linear regression, softmax classification, fully-connected,
convolutional and recurrent neural networks.

Linear regression
~~~~~~~~~~~~~~~~~

Here is the prediction function and the corresponding quadratic loss
function for a simple linear regression model:

::

    predict(w,x) = w[1]*x .+ w[2]

    loss(w,x,y) = sumabs2(y - predict(w,x)) / size(y,2)

The variable ``w`` is a list of parameters (it could be a Tuple,
Array, or Dict), ``x`` is the input and ``y`` is the desired
output. To train this model, we want to adjust its parameters to
reduce the loss on given training examples. The direction in the
parameter space in which the loss reduction is maximum is given by the
negative gradient of the loss. Knet uses the higher-order function
``grad`` from `AutoGrad.jl
<https://github.com/denizyuret/AutoGrad.jl>`__ to compute the gradient
direction:

::

    using Knet

    lossgradient = grad(loss)

Note that ``grad`` is a higher-order function that takes and returns
other functions. The ``lossgradient`` function takes the same arguments
as ``loss``, e.g. ``dw = lossgradient(w,x,y)``. Instead of returning a
loss value, ``lossgradient`` returns ``dw``, the gradient of the loss
with respect to its first argument ``w``. The type and size of ``dw`` is
identical to ``w``, each entry in ``dw`` gives the derivative of the
loss with respect to the corresponding entry in ``w``. See ``@doc grad``
for more information.

Given some training ``data = [(x1,y1),(x2,y2),...]``, here is how we can
train this model:

::

    function train(w, data; lr=.1)
        for (x,y) in data
            dw = lossgradient(w, x, y)
            for i in 1:length(w)
                w[i] -= lr * dw[i]
            end
        end
        return w
    end

We simply iterate over the input-output pairs in data, calculate the
lossgradient for each example, and move the parameters in the negative
gradient direction with a step size determined by the learning rate
``lr``.

.. image:: images/housing.jpeg

Let's train this model on the
`Housing <https://archive.ics.uci.edu/ml/datasets/Housing>`__ dataset
from the UCI Machine Learning Repository.

::

    julia> url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    julia> rawdata = readdlm(download(url))
    julia> x = rawdata[:,1:13]'
    julia> x = (x .- mean(x,2)) ./ std(x,2)
    julia> y = rawdata[:,14:14]'
    julia> w = Any[ 0.1*randn(1,13), 0 ]
    julia> for i=1:10; train(w, [(x,y)]); println(loss(w,x,y)); end
    366.0463078055053
    ...
    29.63709385230451

The dataset has housing related information for 506 neighborhoods in
Boston from 1978. Each neighborhood is represented using 13 attributes
such as crime rate or distance to employment centers. The goal is to
predict the median value of the houses given in $1000's. After
downloading, splitting and normalizing the data, we initialize the
parameters randomly and take 10 steps in the negative gradient
direction. We can see the loss dropping from 366.0 to 29.6. See
`housing.jl
<https://github.com/denizyuret/Knet.jl/blob/master/examples/housing.jl>`__
for more information on this example.

Note that ``grad`` was the only function used that is not in the Julia
standard library. This is typical of models defined in Knet.

Softmax classification
~~~~~~~~~~~~~~~~~~~~~~

In this example we build a simple classification model for the
`MNIST <http://yann.lecun.com/exdb/mnist>`__ handwritten digit
recognition dataset. MNIST has 60000 training and 10000 test examples.
Each input x consists of 784 pixels representing a 28x28 image. The
corresponding output indicates the identity of the digit 0..9.

Classification models handle discrete outputs, as opposed to regression
models which handle numeric outputs. We typically use the cross entropy
loss function in classification models:

::

    function loss(w,x,ygold)
        ypred = predict(w,x)
        ynorm = ypred .- log(sum(exp(ypred),1))
        -sum(ygold .* ynorm) / size(ygold,2)
    end

Other than the change of loss function, the softmax model is identical
to the linear regression model. We use the same ``predict``, same
``train`` and set ``lossgradient=grad(loss)`` as before. To see how well
our model classifies let's define an ``accuracy`` function which returns
the percentage of instances classified correctly:

::

    function accuracy(w, data)
        ncorrect = ninstance = 0
        for (x, ygold) in data
            ypred = predict(w,x)
            ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
            ninstance += size(ygold,2)
        end
        return ncorrect/ninstance
    end

Now let's train a model on the MNIST data:

::

    julia> include(Pkg.dir("Knet/examples/mnist.jl"))
    julia> using MNIST: xtrn, ytrn, xtst, ytst, minibatch
    julia> dtrn = minibatch(xtrn, ytrn, 100)
    julia> dtst = minibatch(xtst, ytst, 100)
    julia> w = Any[ -0.1+0.2*rand(Float32,10,784), zeros(Float32,10,1) ]
    julia> println((:epoch, 0, :trn, accuracy(w,dtrn), :tst, accuracy(w,dtst)))
    julia> for epoch=1:10
               train(w, dtrn; lr=0.5)
               println((:epoch, epoch, :trn, accuracy(w,dtrn), :tst, accuracy(w,dtst)))
           end

    (:epoch,0,:trn,0.11761667f0,:tst,0.121f0)
    (:epoch,1,:trn,0.9005f0,:tst,0.9048f0)
    ...
    (:epoch,10,:trn,0.9196f0,:tst,0.9153f0)

Including ``mnist.jl`` loads the MNIST data, downloading it from the
internet if necessary, and provides a training set (xtrn,ytrn), test set
(xtst,ytst) and a ``minibatch`` utility which we use to rearrange the
data into chunks of 100 instances. After randomly initializing the
parameters we train for 10 epochs, printing out training and test set
accuracy at every epoch. The final accuracy of about 92% is close to the
limit of what we can achieve with this type of model. To improve further
we must look beyond linear models.

Multi-layer perceptron
~~~~~~~~~~~~~~~~~~~~~~

A multi-layer perceptron, i.e. a fully connected feed-forward neural
network, is basically a bunch of linear regression models stuck together
with non-linearities in between. We can define one by slightly modifying
the predict function:

::

    function predict(w,x)
        for i=1:2:length(w)-2
            x = max(0, w[i]*x .+ w[i+1])
        end
        return w[end-1]*x .+ w[end]
    end

Here ``w[2k-1]`` is the weight matrix and ``w[2k]`` is the bias vector
for the k'th layer. max(0,a) implements the popular rectifier
non-linearity. Note that if w only has two entries, this is equivalent
to the linear and softmax models. By adding more entries to w, we can
define multi-layer perceptrons of arbitrary depth. Let's define one with
a single hidden layer of 64 units:

::

    w = Any[ -0.1+0.2*rand(Float32,64,784), zeros(Float32,64,1),
             -0.1+0.2*rand(Float32,10,64),  zeros(Float32,10,1) ]

The rest of the code is the same as the softmax model. We use the same
cross-entropy loss function and the same training script. The code for
this example is available in
`mnist.jl <https://github.com/denizyuret/Knet.jl/blob/master/examples/mnist.jl>`__.
The multi-layer perceptron does significantly better than the softmax
model:

::

    (:epoch,0,:trn,0.10166667f0,:tst,0.0977f0)
    (:epoch,1,:trn,0.9389167f0,:tst,0.9407f0)
    ...
    (:epoch,10,:trn,0.9866f0,:tst,0.9735f0)

Convolutional neural network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To improve the performance further, we can use `convolutional neural
networks <http://cs231n.github.io/convolutional-networks/>`__. We will
implement the `LeNet <http://yann.lecun.com/exdb/lenet>`__ model which
consists of two convolutional layers followed by two fully connected
layers. Knet provides the ``conv4(w,x)`` and ``pool(x)`` functions for
the implementation of convolutional nets (see ``@doc conv4`` and
``@doc pool`` for more information):

::

    function predict(w,x0)
        x1 = pool(max(0, conv4(w[1],x0) .+ w[2]))
        x2 = pool(max(0, conv4(w[3],x1) .+ w[4]))
        x3 = max(0, w[5]*mat(x2) .+ w[6])
        return w[7]*x3 .+ w[8]
    end

The weights for the convolutional net can be initialized as follows:

::

    w = Any[ -0.1+0.2*rand(Float32,5,5,1,20),  zeros(Float32,1,1,20,1),
             -0.1+0.2*rand(Float32,5,5,20,50), zeros(Float32,1,1,50,1),
             -0.1+0.2*rand(Float32,500,800),   zeros(Float32,500,1),
             -0.1+0.2*rand(Float32,10,500),    zeros(Float32,10,1) ]

Currently convolution and pooling are only supported on the GPU for 4-D
and 5-D arrays. So we reshape our data and transfer it to the GPU along
with the parameters by converting them into KnetArrays (see
``@doc KnetArray`` for more information):

::

    dtrn = map(d->(KnetArray(reshape(d[1],(28,28,1,100))), KnetArray(d[2])), dtrn)
    dtst = map(d->(KnetArray(reshape(d[1],(28,28,1,100))), KnetArray(d[2])), dtst)
    w = map(KnetArray, w)

The training proceeds as before giving us even better results. The code
for the LeNet example can be found in
`lenet.jl <https://github.com/denizyuret/Knet.jl/blob/master/examples/lenet.jl>`__.

::

    (:epoch,0,:trn,0.12215f0,:tst,0.1263f0)
    (:epoch,1,:trn,0.96963334f0,:tst,0.971f0)
    ...
    (:epoch,10,:trn,0.99553335f0,:tst,0.9879f0)

Recurrent neural network
~~~~~~~~~~~~~~~~~~~~~~~~

In this section we will see how to implement a recurrent neural network
(RNN) in Knet. An RNN is a class of neural network where connections
between units form a directed cycle, which allows them to keep a
persistent state over time. This gives them the ability to process
sequences of arbitrary length one element at a time, while keeping track
of what happened at previous elements. As an example, we will build a
character-level language model inspired by `"The Unreasonable
Effectiveness of Recurrent Neural
Networks" <http://karpathy.github.io/2015/05/21/rnn-effectiveness>`__
from the Andrej Karpathy blog. The model can be trained with different
genres of text, and can be used to generate original text in the same
style.

It turns out simple RNNs are not very good at remembering things for a
very long time. Currently the most popular solution is to use a more
complicated unit like the Long Short Term Memory (LSTM). An LSTM
controls the information flow into and out of the unit using gates
similar to digital circuits and can model long term dependencies. See
`Understanding LSTM
Networks <http://colah.github.io/posts/2015-08-Understanding-LSTMs>`__
by Christopher Olah for a good overview of LSTMs.

The code below shows one way to define an LSTM in Knet. The first two
arguments are the parameters, the weight matrix and the bias vector. The
next two arguments hold the internal state of the LSTM: the hidden and
cell arrays. The last argument is the input. Note that for performance
reasons we lump all the parameters of the LSTM into one matrix-vector
pair instead of using separate parameters for each gate. This way we can
perform a single matrix multiplication, and recover the gates using
array indexing. We represent input, hidden and cell as row vectors
rather than column vectors for more efficient concatenation and
indexing. ``sigm`` and ``tanh`` are the sigmoid and the hyperbolic
tangent activation functions. The LSTM returns the updated state
variables ``hidden`` and ``cell``.

::

    function lstm(weight,bias,hidden,cell,input)
        gates   = hcat(input,hidden) * weight .+ bias
        hsize   = size(hidden,2)
        forget  = sigm(gates[:,1:hsize])
        ingate  = sigm(gates[:,1+hsize:2hsize])
        outgate = sigm(gates[:,1+2hsize:3hsize])
        change  = tanh(gates[:,1+3hsize:end])
        cell    = cell .* forget + ingate .* change
        hidden  = outgate .* tanh(cell)
        return (hidden,cell)
    end

The LSTM has an input gate, forget gate and an output gate that control
information flow. Each gate depends on the current ``input`` value, and
the last hidden state ``hidden``. The memory value ``cell`` is computed
by blending a new value ``change`` with the old ``cell`` value under the
control of input and forget gates. The output gate decides how much of
the ``cell`` is shared with the outside world.

If an input gate element is close to 0, the corresponding element in the
new ``input`` will have little effect on the memory cell. If a forget
gate element is close to 1, the contents of the corresponding memory
cell can be preserved for a long time. Thus the LSTM has the ability to
pay attention to the current input, or reminisce in the past, and it can
learn when to do which based on the problem.

To build a language model, we need to predict the next character in a
piece of text given the current character and recent history as encoded
in the internal state. The ``predict`` function below implements a
multi-layer LSTM model. ``s[2k-1:2k]`` hold the hidden and cell arrays
and ``w[2k-1:2k]`` hold the weight and bias parameters for the k'th LSTM
layer. The last three elements of ``w`` are the embedding matrix and the
weight/bias for the final prediction. ``predict`` takes the current
character encoded in ``x`` as a one-hot row vector, multiplies it with
the embedding matrix, passes it through a number of LSTM layers, and
converts the output of the final layer to the same number of dimensions
as the input using a linear transformation. The state variable ``s`` is
modified in-place.

::

    function predict(w, s, x)
        x = x * w[end-2]
        for i = 1:2:length(s)
            (s[i],s[i+1]) = lstm(w[i],w[i+1],s[i],s[i+1],x)
            x = s[i]
        end
        return x * w[end-1] .+ w[end]
    end

To train the language model we will use Backpropagation Through Time
(BPTT) which basically means running the network on a given sequence and
updating the parameters based on the total loss. Here is a function that
calculates the total cross-entropy loss for a given (sub)sequence:

::

    function loss(param,state,sequence,range=1:length(sequence)-1)
        total = 0.0; count = 0
        atype = typeof(getval(param[1]))
        input = convert(atype,sequence[first(range)])
        for t in range
            ypred = predict(param,state,input)
            ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
            ygold = convert(atype,sequence[t+1])
            total += sum(ygold .* ynorm)
            count += size(ygold,1)
            input = ygold
        end
        return -total / count
    end

Here ``param`` and ``state`` hold the parameters and the state of the
model, ``sequence`` and ``range`` give us the input sequence and a
possible range over it to process. We convert the entries in the
sequence to inputs that have the same type as the parameters one at a
time (to conserve GPU memory). We use each token in the given range as
an input to predict the next token. The average cross-entropy loss per
token is returned.

To generate text we sample each character randomly using the
probabilities predicted by the model based on the previous character:

::

    function generate(param, state, vocab, nchar)
        index_to_char = Array(Char, length(vocab))
        for (k,v) in vocab; index_to_char[v] = k; end
        input = oftype(param[1], zeros(1,length(vocab)))
        index = 1
        for t in 1:nchar
            ypred = predict(param,state,input)
            input[index] = 0
            index = sample(exp(logp(ypred)))
            print(index_to_char[index])
            input[index] = 1
        end
        println()
    end

Here ``param`` and ``state`` hold the parameters and state variables as
usual. ``vocab`` is a Char->Int dictionary of the characters that can be
produced by the model, and ``nchar`` gives the number of characters to
generate. We initialize the input as a zero vector and use ``predict``
to predict subsequent characters. ``sample`` picks a random index based
on the normalized probabilities output by the model.

At this point we can train the network on any given piece of text (or
other discrete sequence). For efficiency it is best to minibatch the
training data and run BPTT on small subsequences. See
`charlm.jl <https://github.com/denizyuret/Knet.jl/blob/master/examples/charlm.jl>`__
for details. Here is a sample run on 'The Complete Works of William
Shakespeare':

::

    $ cd .julia/Knet/examples
    $ wget http://www.gutenberg.org/files/100/100.txt
    $ julia charlm.jl --data 100.txt --epochs 10 --winit 0.3 --save shakespeare.jld
    ... takes about 10 minutes on a GPU machine
    $ julia charlm.jl --load shakespeare.jld --generate 1000

        Pand soping them, my lord, if such a foolish?
      MARTER. My lord, and nothing in England's ground to new comp'd.
        To bless your view of wot their dullst. If Doth no ape;
        Which with the heart. Rome father stuff
        These shall sweet Mary against a sudden him
        Upon up th' night is a wits not that honour,
        Shouts have sure?
      MACBETH. Hark? And, Halcance doth never memory I be thou what
        My enties mights in Tim thou?
      PIESTO. Which it time's purpose mine hortful and
        is my Lord.
      BOTTOM. My lord, good mine eyest, then: I will not set up.
      LUCILIUS. Who shall

Under the hood
--------------

Coming soon...

Benchmarks
----------

Coming soon...

Contributing
------------

Knet is an open-source project and we are always open to new
contributions: bug reports and fixes, feature requests and
contributions, new machine learning models and operators, inspiring
examples, benchmarking results are all welcome. If you need help or
would like to request a feature, please consider joining the
`knet-users <https://groups.google.com/forum/#!forum/knet-users>`__
mailing list. If you find a bug, please open a `GitHub
issue <https://github.com/denizyuret/Knet.jl/issues>`__. If you would
like to contribute to Knet development, check out the
`knet-dev <https://groups.google.com/forum/#!forum/knet-dev>`__ mailing
list and `tips for
developers <http://knet.readthedocs.org/en/latest/install.html#tips-for-developers>`__.
If you use Knet in your own work, the suggested citation is:

::

    @misc{knet,
      author={Yuret, Deniz},
      title={Knet: Ko\c{c} University deep learning framework.},
      year={2016},
      howpublished={\url{https://github.com/denizyuret/Knet.jl}}
    }

