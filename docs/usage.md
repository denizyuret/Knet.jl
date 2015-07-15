## Usage

We will use the [MNIST](http://yann.lecun.com/exdb/mnist) dataset to
illustrate basic usage of
[KUnet](https://github.com/denizyuret/KUnet.jl):

```
julia> require(Pkg.dir("KUnet/test/mnist.jl"))
```

This may take a bit the first time you run to download the data.

Next we tell Julia we intend to use KUnet, and some variables from
MNIST:

```
julia> using KUnet
julia> using MNIST: xtrn, ytrn, xtst, ytst
```

The MNIST variables are Float32 matrices.  The x matrices have pixel
values scaled to [0.0:1.0] for a 28x28 image on each column.  The y
matrices have 10 rows indicating the 10 classes with a single nonzero
entry for the correct class in each column.

```
julia> xtrn, ytrn, xtst, ytst
(
784x60000 Array{Float32,2}: ...
10x60000 Array{Float32,2}: ...
784x10000 Array{Float32,2}: ...
10x10000 Array{Float32,2}: ...
)
```

Let's construct a neural net with a single layer of 64 hidden units
using the relu activation function and the cross entropy loss function.

```
julia> net = [ Mmul(64), Bias(), Relu(),
               Mmul(10),  Bias(), XentLoss() ]
```

Each element of the net array represents an operation, e.g. Mmul
multiplies its input with a weight matrix, Bias adds a bias vector,
Relu applies the rectified linear transformation to each element etc.
They are subtypes of an abstract type called Layer.  The full list of
Layers currently implemented are described in [Layers](layers.md).

A Net is simply a 1-D array of Layers.  Here are the definitions from
[net.jl](https://github.com/denizyuret/KUnet.jl/blob/master/src/net.jl) and 
[bias.jl](https://github.com/denizyuret/KUnet.jl/blob/master/src/bias.jl):

```
abstract Layer
typealias Net Array{Layer,1}
type Bias <: Layer; b; Bias()=new(Param(0)); end
```

If you are not happy with the default Layer constructors, you can
specify your own parameters.  For example, the default Mmul(64)
constructor fills a (64,784) weight matrix with random weights from a
Gaussian distribution with mean=0 and std=0.01.  If we want a
different initialization, we could create a weight matrix any way we
want and pass it to the Mmul constructor instead.  Note that the
weight matrix for an Mmul layer with 784 inputs and 64 outputs has
size (64, 784).

```
julia> w1 = float32(randn(64, 784) * 0.05)
julia> l1 = Mmul(w1)
```

Training parameters like the learning rate (lr) can be specified at
layer construction, or using setparam! on the whole network or on
individual layers.

```
julia> l1 = Mmul(64; lr=0.01)
julia> setparam!(l1; lr=0.01)
julia> setparam!(net; lr=0.01)
```

It is also possible to save nets to
[JLD](https://github.com/timholy/HDF5.jl) files using
`savenet(fname::String, n::Net)` and read them using
`loadnet(fname::String)`.  Let's save our initial random network for
replicatibility.

```
julia> savenet("net0.jld", net)
```

OK, now that we have some data and a network, let's proceed with training.

```
@time for i=1:100
    train(net, xtrn, ytrn; batch=128)
    println((i, accuracy(ytst, predict(net, xtst)), 
                accuracy(ytrn, predict(net, xtrn))))
end
```

If you take a look at the definition of `train` in
[net.jl](https://github.com/denizyuret/KUnet.jl/blob/master/src/net.jl),
you will see that it takes a network net, the input x, and the desired
output y, and after splitting the data into minibatches of size
`batch`, it just calls `backprop` and `update`.  Here is a simplified
version:

```
train(net, x, y)=(backprop(net, x, y); update(net))
```

The `backprop` function calls `forw` which computes the network
output, and `back` which computes the gradients of the network
parameters with respect to the loss function:

```
backprop(net, x, y)=(forw(net, x); back(net, y))
```

The `forw` and `back` functions for a Net simply call the `forw` and
`back` functions of each layer in order, feeding the output of one to
the input of the next:

```
forw(n::Net, x)=(for i=1:length(n); x=forw(n[i], x); end)
back(n::Net, y)=(for i=length(n):-1:1; y=back(n[i], y); end)
```

The `forw` function for a layer takes the layer input x, and returns
the layer output y.  The `back` function for a layer takes the loss
gradient wrt its output dy and returns the loss gradient wrt its input
dx.  If the layer has a parameter w, `back` also computes the loss
gradient w.diff wrt its current value w.data.  You can take a look at
individual layer definitions (e.g. in
[mmul.jl](https://github.com/denizyuret/KUnet.jl/blob/master/src/mmul.jl),
[bias.jl](https://github.com/denizyuret/KUnet.jl/blob/master/src/bias.jl),
[relu.jl](https://github.com/denizyuret/KUnet.jl/blob/master/src/relu.jl),
etc.) to see how this is done for each layer.

The final layer of the network
([XentLoss](https://github.com/denizyuret/KUnet.jl/blob/master/src/xentloss.jl)
in our case) is a subtype of LossLayer.  LossLayer is a special type
of layer: its forw does nothing but record the network output y.  Its
back expects the desired output z (not a gradient) and computes the
loss gradient wrt the network output dy.  A LossLayer also implements
the function `loss(l::LossLayer,z)` which returns the actual loss
value given the desired output z.  See [Loss Layers](loss.md) for a
complete list of loss layers implemented in KUnet.

The `update` function for a net calls the `update` function for each
of its layers, which in turn calls the `update` function on layer
parameters:

```
update(n::Net)=(for l in n; update(l); end)
update(l::Bias)=update(l.b)
```

The `update` function for a parameter p is used to update its values
(p.data) given the loss gradients (p.diff).  Its behavior is
controlled by the following parameters: lr, l1reg, l2reg, adagrad,
momentum, nesterov.  Here is a simplified definition of `update` from
[param.jl](https://github.com/denizyuret/KUnet.jl/blob/master/src/param.jl)
(p.ada, p.mom, and p.nes are temporary arrays initialized to 0):

```
function update(p::Param; o...)
    isdefined(p,:l1reg)    && (p.diff += p.l1reg * sign(p.data))
    isdefined(p,:l2reg)    && (p.diff += p.l2reg * p.data)
    isdefined(p,:adagrad)  && (p.ada += p.diff.^2; p.diff /= p.adagrad + sqrt(p.ada))
    isdefined(p,:momentum) && (p.diff += p.momentum * p.mom; p.mom[:] = p.diff)
    isdefined(p,:nesterov) && (p.nes *= p.nesterov; p.nes += p.diff; p.diff += p.nesterov * p.nes)
    isdefined(p,:lr)       && (p.diff *= p.lr)
    p.data -= p.diff
end
```

Our training should print out the test set and training set accuracy
at the end of every epoch.

```
(1,0.3386,0.3356)
(2,0.7311,0.7226666666666667)
(3,0.821,0.8157333333333333)
...
(99,0.9604,0.9658166666666667)
(100,0.9604,0.96605)
elapsed time: 39.738191211 seconds (1526525108 bytes allocated, 3.05% gc time)
```

Note that for actual research we should not be looking at the test set
accuracy at this point.  We should instead split the training set into
a training and a development portion and do all our playing around
with those.  We should also run each experiment 10 times with
different random seeds and measure standard errors, etc.  But, this is
just a KUnet tutorial.

It seems the training set accuracy is not that great.  Maybe
increasing the learning rate may help:

```
net = loadnet("net0.jld")
setparam!(net, lr=0.5)

# same for loop...

(1,0.9152,0.9171833333333334)
(2,0.9431,0.9440333333333333)
(3,0.959,0.9611666666666666)
...
(59,0.9772,0.9999833333333333)
(60,0.9773,1.0)
...
(100,0.9776,1.0)
```

Wow!  We got 100% training set accuracy in 60 epochs.  This should
drive home the importance of setting a good learning rate.

But the test set is still lagging behind.  What if we try increasing
the number of hidden units:

```
for h in (128, 256, 512, 1024)
    net = [Mmul(h), Bias(), Relu(), Mmul(10),  Bias(), XentLoss()]
    setparam!(net; lr=0.5)
    for i=1:100
        train(net, xtrn, ytrn; batch=128)
        println((i, accuracy(ytst, predict(net, xtst)), 
                    accuracy(ytrn, predict(net, xtrn))))
    end
end

# Number of epochs and test accuracy when training accuracy reaches 1.0:
# 128:  (43,0.9803,1.0)
# 256:  (42,0.983,1.0)
# 512:  (36,0.983,1.0)
# 1024: (30,0.9833,1.0)
```

This improvement is unexpected, we were already overfitting with 64
hidden units, and common wisdom is not to increase the capacity of the
network by increasing the hidden units in that situation.  Maybe we
should try [dropout](http://jmlr.org/papers/v15/srivastava14a.html):

```
net = [Drop(0.2), Mmul(1024), Bias(), Relu(), 
       Drop(0.5), Mmul(10),  Bias(), XentLoss()]

# lr=0.5, same for loop
...
(100,0.9875,0.9998166666666667)
elapsed time: 122.898730432 seconds (1667849932 bytes allocated, 0.96% gc time)
```

Or bigger and bigger nets:

```
net = [Drop(0.2), Mmul(4096),  Bias(), Relu(), 
       Drop(0.5), Mmul(4096), Bias(), Relu(), 
       Drop(0.5), Mmul(10),   Bias(), XentLoss()]

# lr=0.5, same for loop
...
(100,0.9896,0.9998166666666667)
elapsed time: 804.242212488 seconds (1080 MB allocated, 0.02% gc time in 49 pauses with 0 full sweep)
```

Or maybe we should try convolution.  Here is an implementation of
[LeNet](http://yann.lecun.com/exdb/lenet):

```
net = [Conv(20,5), Bias(), Relu(), Pool(2),
       Conv(50,5), Bias(), Relu(), Pool(2),
       Mmul(500), Bias(), Relu(),
       Mmul(10), Bias(), XentLoss()]
setparam!(net; lr=0.1)

# Need to reshape the input arrays for convolution:
xtrn2 = reshape(xtrn, 28, 28, 1, size(xtrn, 2))
xtst2 = reshape(xtst, 28, 28, 1, size(xtst, 2))

# same for loop
...
(100,0.9908,1.0)
elapsed time: 360.722851006 seconds (5875158944 bytes allocated, 1.95% gc time)
```

OK, that's enough fiddling around.  I hope this gave you enough to get
your hands dirty.  We are already among the better results on the
[MNIST website](http://yann.lecun.com/exdb/mnist).  I am sure you can
do better playing around with the learning rate, dropout
probabilities, momentum, adagrad, regularization, and numbers, sizes,
types of layers etc.  But be careful, it could become addictive :)
