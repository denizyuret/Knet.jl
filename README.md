# Knet

[![Build Status](https://travis-ci.org/denizyuret/Knet.jl.svg?branch=master)](https://travis-ci.org/denizyuret/Knet.jl)
<!-- 
TODO: https://github.com/JuliaCI/Coverage.jl
[![Coverage Status](https://coveralls.io/repos/denizyuret/Knet.jl/badge.svg)](https://coveralls.io/r/denizyuret/Knet.jl)
[![Knet](http://pkg.julialang.org/badges/Knet_0.3.svg)](http://pkg.julialang.org/?pkg=Knet)
[![Knet](http://pkg.julialang.org/badges/Knet_0.4.svg)](http://pkg.julialang.org/?pkg=Knet)
[![Knet](http://pkg.julialang.org/badges/Knet_0.5.svg)](http://pkg.julialang.org/?pkg=Knet)
-->

[Knet](http://knet.rtfd.org) (pronounced "kay-net") is the [Koç
University](http://www.ku.edu.tr) deep learning framework implemented
in [Julia](http://julia.rtfd.org) by [Deniz
Yuret](http://www.denizyuret.com) and collaborators.  It supports
construction of high-performance deep learning models in plain Julia
by combining automatic differentiation with efficient GPU kernels and
memory management.  Models can be defined and trained using arbitrary
Julia code with helper functions, loops, conditionals, recursion,
closures, array indexing and concatenation.  The training can be
performed on the GPU by simply using KnetArray instead of Array for
parameters and data.  Check out the
[tutorial](http://knet.readthedocs.io/en/latest/intro.html),
[examples](https://github.com/denizyuret/Knet.jl/tree/master/examples),
and [documentation](http://knet.rtfd.org) for more information.

## <a name="cont"></a> Contents

* [Installation](#inst)
* [Examples](#exam)
  - [Linear regression](#line)
  - [Softmax classification](#soft)
  - [Multi-layer perceptron](#mult)
  - [Convolutional neural network](#conv)
  - [Recurrent neural network](#recu)
* [Under the hood](#unde)
* [Benchmarks](#benc)
* [See also](#seea)


## <a name="inst"></a> Installation

You can install Knet using `Pkg.add("Knet")`.  Some of the examples
use additional packages such as ArgParse, GZip, and CUDNN.  These are
not required by Knet and can be installed when needed using additional
`Pkg.add()` commands.  The documentation provides detailed
[installation
instructions](http://knet.readthedocs.org/en/dev/install.html#installation)
as well as a section on [using Amazon
AWS](http://knet.readthedocs.org/en/dev/install.html#using-amazon-aws)
to experiment with GPU machines on the cloud with pre-installed Knet
images.

## <a name="exam"></a> Examples

In Knet, a machine learning model is defined using plain Julia code.
A typical model consists of a prediction and a loss function.  The
prediction function takes model parameters and some input, returns the
prediction of the model for that input.  The loss function measures
how bad the prediction is with respect to some desired output.

### <a name="line"></a> Linear regression

Here is the prediction function and the corresponding quadratic loss
function for a simple linear regression model:

```
using Knet

predict(w,x) = w[1]*x .+ w[2]

loss(w,x,y) = sumabs2(y - predict(w,x)) / size(y,2)
```

`w` is a list of parameters (it could be a Tuple, Array, or Dict), `x`
is the input and `y` is the desired output.  To train this model, we
want to adjust its parameters to reduce the loss on some training
examples.  The direction in the parameter space in which the loss
reduction is maximum is the negative gradient of the loss.  Knet uses
the higher-order function `grad` from
[AutoGrad.jl](https://github.com/denizyuret/AutoGrad.jl) to compute
the gradient direction:

```
lossgradient = grad(loss)
```

Note that `grad` is a higher-order function that takes and returns
other functions.  The `lossgradient` function takes the same arguments
as `loss`, e.g. `dw = lossgradient(w,x,y)`.  Instead of returning a
loss, it returns `dw`, the gradient of the loss with respect to its
first argument `w`.  `dw` has the same type and size as `w`.  Each
entry in `dw` gives the derivative of the loss with respect to that
entry in `w`.

Given some training `data = [(x1,y1),(x2,y2),...]`, here is how we can
train this model:

```
function train(w, data; lr=.1)
    for (x,y) in data
        dw = lossgradient(w, x, y)
        for i in 1:length(w)
            w[i] -= lr * dw[i]
        end
    end
    return w
end
```

We simply iterate over the input-output pairs in data, calculate the
lossgradient for each example, and move the parameters in the opposite
direction with a step size determined by the learning rate `lr`.

Let's train this model on the
[Housing](https://archive.ics.uci.edu/ml/datasets/Housing) dataset
from the UCI Machine Learning Repository.  

```
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
```

The dataset has housing related information for 506 neighborhoods in
Boston from 1978.  Each neighborhood is represented using 13
attributes such as crime rate, and the goal is to predict the median
dollar value of the houses.  After downloading, splitting and
normalizing the data, we initialize the parameters randomly and take
10 steps in the negative gradient direction.  We can see the loss
dropping from 366.0 to 29.6.  See
[housing.jl](https://github.com/denizyuret/Knet.jl/blob/master/examples/housing.jl)
for more information on this example.

### <a name="soft"></a> Softmax classification

In this example we build a simple classification model for the
[MNIST](http://yann.lecun.com/exdb/mnist) handwritten digit
recognition dataset.  MNIST has 60000 training and 10000 test
examples. Each input x consists of 784 pixels representing a 28x28
image.

Classification models handle discrete outputs (in this case the
identity of the digit 0..9), as opposed to regression models, which
handle numeric outputs.  We typically use the cross entropy loss
function in classification models:

```
function loss(w,x,ygold)
    ypred = predict(w,x)
    ynorm = ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end
```

Other than the change of loss function, the softmax model is identical
to the linear regression model.  We use the same `predict`, same
`train` and set `lossgradient=grad(loss)` as before.  To see how well
our model classifies we can use an `accuracy` function which returns
the percentage of instances classified correctly:

```
function accuracy(w, data)
    ncorrect = ninstance = 0
    for (x, ygold) in data
        ypred = predict(w,x)
        ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
        ninstance += size(ygold,2)
    end
    return ncorrect/ninstance
end
```

Now let's train a model on the MNIST data:

```
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
```

Including `mnist.jl` loads the MNIST data, downloading from the
internet if necessary, and provides a training set (xtrn,ytrn), test
set (xtst,ytst) and a `minibatch` utility which we use to rearrange
the data into chunks of 100 instances.  After randomly initializing
the parameters we train for 10 epochs, printing out training and test
set accuracy at every epoch.  The final accuracy of about 92% is close
to the limit of what we can achieve with this type of model.  To
improve further we must look beyond linear models.


### <a name="mult"></a> Multi-layer perceptron

A multi-layer perceptron, i.e. a fully connected feed-forward neural
network, is basically a bunch of linear regression models stuck
together with non-linearities in between.  We can define one by
slightly modifying the predict function:

```
function predict(w,x)
    for i=1:2:length(w)-2
        x = max(0, w[i]*x .+ w[i+1])
    end
    return w[end-1]*x .+ w[end]
end
```

Here `w[2k-1]` is the weight matrix and `w[2k]` is the bias vector for
the k'th layer.  max(0,a) implements the popular rectifier
non-linearity.  Note that if w only has two entries, this is
equivalent to the linear and softmax models.  By adding more entries to
w, we can define multi-layer perceptrons of arbitrary depth.  Let's
define one with a single hidden layer of 64 units:

```
w = Any[ -0.1+0.2*rand(Float32,64,784), zeros(Float32,64,1),
         -0.1+0.2*rand(Float32,10,64),  zeros(Float32,10,1) ]
```

The rest of the code is the same as the softmax model.  We use the
same cross-entropy loss function and the same training script.  The
code for this example is available in
[mnist.jl](https://github.com/denizyuret/Knet.jl/blob/master/examples/mnist.jl).
The multi-layer perceptron does significantly better than the softmax
model:

```
(:epoch,0,:trn,0.10166667f0,:tst,0.0977f0)
(:epoch,1,:trn,0.9389167f0,:tst,0.9407f0)
...
(:epoch,10,:trn,0.9866f0,:tst,0.9735f0)
```

### <a name="conv"></a> Convolutional neural network

To improve the performance further, we can use [convolutional neural
networks](http://cs231n.github.io/convolutional-networks/).  We will
implement the [LeNet](http://yann.lecun.com/exdb/lenet) model which
consists of two convolutional layers followed by two fully connected
layers.  Knet provides the `conv4(w,x)` and `pool(x)` functions for
the implementation of convolutional nets:

```
function predict(w,x0)
    x1 = pool(max(0, conv4(w[1],x0) .+ w[2]))
    x2 = pool(max(0, conv4(w[3],x1) .+ w[4]))
    x3 = max(0, w[5]*mat(x2) .+ w[6])
    return w[7]*x3 .+ w[8]
end
```

The weights for the convolutional net can be initialized as follows:

```
w = Any[ -0.1+0.2*rand(Float32,5,5,1,20),  zeros(Float32,1,1,20,1),
         -0.1+0.2*rand(Float32,5,5,20,50), zeros(Float32,1,1,50,1),
         -0.1+0.2*rand(Float32,500,800),   zeros(Float32,500,1),
         -0.1+0.2*rand(Float32,10,500),    zeros(Float32,10,1) ]
```

Currently convolution and pooling are only supported on the GPU for
4-D and 5-D arrays.  So we reshape our data and transfer it to the GPU
along with the parameters by converting them into KnetArrays:

```
dtrn = map(d->(KnetArray(reshape(d[1],(28,28,1,100))), KnetArray(d[2])), dtrn)
dtst = map(d->(KnetArray(reshape(d[1],(28,28,1,100))), KnetArray(d[2])), dtst)
w = map(KnetArray, w)
```

The training proceeds as before giving us even better results:

```
(:epoch,0,:trn,0.12215f0,:tst,0.1263f0)
(:epoch,1,:trn,0.96963334f0,:tst,0.971f0)
...
(:epoch,10,:trn,0.99553335f0,:tst,0.9879f0)
```

### <a name="recu"></a> Recurrent neural network


## <a name="unde"></a> Under the hood


## <a name="benc"></a> Benchmarks


## <a name="seea"></a> See also

* If you would like a quick introduction to Knet, try the [tutorial](http://knet.readthedocs.org/en/latest/intro.html).
* If you would like to try Knet on your own computer, please follow the [installation instructions](http://knet.readthedocs.org/en/dev/install.html#installation).
* If you would like to try working with a GPU and do not have access to one, take a look at the [using Amazon AWS](http://knet.readthedocs.org/en/dev/install.html#using-amazon-aws) tutorial.
* If you need help or would like to request a feature, please consider joining the [knet-users](https://groups.google.com/forum/#!forum/knet-users) mailing list.
* If you find a bug, please open a [GitHub issue](https://github.com/denizyuret/Knet.jl/issues).  
* If you would like to contribute to Knet development, check out the [knet-dev](https://groups.google.com/forum/#!forum/knet-dev) mailing list and [tips for developers](http://knet.readthedocs.org/en/dev/install.html#tips-for-developers).

Knet is an open-source project and we are always open to new
contributions: bug reports and fixes, feature requests and
contributions, new machine learning models and operators, inspiring
examples, benchmarking results are all welcome.  If you use Knet in
your own work, the suggested citation is:

```
@misc{knet,
  author={Yuret, Deniz},
  title={Knet: Ko\c{c} University deep learning framework.},
  year={2016},
  howpublished={\url{https://github.com/denizyuret/Knet.jl}}
}
```
