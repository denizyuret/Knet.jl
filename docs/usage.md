## Usage

We will use the MNIST dataset to illustrate basic usage of KUnet:
```
julia> include(Pkg.dir("KUnet/test/mnist.jl"))
```

This may take a bit the first time you run to download the data.

Next we tell Julia we intend to use KUnet, and some variables from MNIST:
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
using the relu activation function.
```
julia> net = Net(relu, 784, 64, 10);
```

We could have done the same thing constructing each layer separately.
The two constructions are exactly equivalent:

```
julia> l1 = Layer(relu, 784, 64);
julia> l2 = Layer(64, 10);
julia> net = [l1, l2];
```

Note that the last layer has no activation function, we just need the
linear output for classification.

By default the Layer constructor picks random weights from a Gaussian
distribution with std=0.01 and zero bias vectors.  We could create a
weight matrix any way we want and pass it to the Layer constructor
instead:

```
julia> w1 = float32(randn(64, 784) * 0.05)
julia> b1 = zeros(Float32, 64, 1)
julia> l1 = Layer(relu, w1, b1)
```

Note that the weight matrix for a Layer with 784 inputs and 64 outputs
has size (64, 784).

The bias, as well as the activation function, are optional.  Training
parameters like the learningRate can be specified during net/layer
construction, as well as afterward using setparam!.  It is also
possible to save layers to HDF5 files using `h5write(fname::String,
l::Layer)` and read them using `Layer(fname::String)`.  Please see
`types.jl` and `h5io.jl` for details.

OK, now that we have some data and a network, let's proceed with training.
Here is a convenience function to measure the classification accuracy:
```
julia> accuracy(y,z)=mean(findmax(y,1)[2] .== findmax(z,1)[2])
```

Let's do 100 epochs using default settings (learningRate=0.01, minibatch size=128):
```
for i=1:100
    train(net, xtrn, ytrn)
    println((i, accuracy(ytst, predict(net, xtst)), 
                accuracy(ytrn, predict(net, xtrn))))
end
```

This should print out the test set and training set accuracy at the end of
every epoch.  100 epochs take about 26 seconds with a K20 GPU:
```
(1,0.3665,0.36438333333333334)
(2,0.7304,0.7236166666666667)
(3,0.8264,0.82115)
...
(99,0.9616,0.9666)
(100,0.9619,0.9668833333333333)
```

Note that for actual research we should not be looking at the test set 
accuracy at this point.  We should instead split the training set into a training and a development portion and do all our playing around with those.  We should also run each experiment 10 times with different random seeds and measure standard errors, etc.  But, this is just a KUnet tutorial.

It seems the training set accuracy is not that great.  Maybe increasing the learning rate may help:
```
julia> net = Net(relu, 784, 64, 10)
julia> setparam!(net, learningRate=0.5)
for i=1:100
    train(net, xtrn, ytrn)
    println((i, accuracy(ytst, predict(net, xtst)), 
                accuracy(ytrn, predict(net, xtrn))))
end

(1,0.9112,0.91185)
(2,0.9441,0.9436)
(3,0.9579,0.9598666666666666)
...
(50,0.9791,0.9999833333333333)
(51,0.9793,1.0)
```

Wow!  We got 100% training set accuracy in 50 epochs.  Let's save this network for posterity:
```
julia> h5write("layer1.h5", net[1])
julia> h5write("layer2.h5", net[2])
```

But the test set is still lagging behind.  What if we try increasing the number of hidden units (use the same for loop for each net below):
```
julia> net = Net(relu, 784, 128, 10; learningRate=0.5)  # (44,0.9808,1.0)
julia> net = Net(relu, 784, 256, 10; learningRate=0.5)  # (37,0.9827,1.0)
julia> net = Net(relu, 784, 512, 10; learningRate=0.5)  # (35,0.983,1.0)
julia> net = Net(relu, 784, 1024, 10; learningRate=0.5)  # (30,0.9835,1.0)
```

This is unexpected, we were already overfitting with 64 hidden units, and common wisdom is not to increase the capacity of the network by increasing the hidden units in that situation.  Maybe we should try dropout:
```
julia> net = Net(relu, 784, 1024, 10; dropout=0.5, learningRate=0.5)
julia> setparam!(net[1], dropout=0.2)   # first layer drops less
@time for i=1:100                                                                                                   
    train(net, xtrn, ytrn)                                                                                                 
    println((i, accuracy(ytst, predict(net, xtst)), accuracy(ytrn, predict(net, xtrn))))                                   
end
...
(100,0.988,0.9999)
elapsed time: 70.73067047 seconds (875 MB allocated, 0.18% gc time in 40 pauses with 0 full sweep)
```

Or bigger and bigger nets:
```
julia> net = Net(relu, 784, 4096, 4096, 10; dropout=0.5, learningRate=0.5)
julia> setparam!(net[1], dropout=0.2)
# same for loop...
(100,0.9896,0.9998166666666667)
elapsed time: 804.242212488 seconds (1080 MB allocated, 0.02% gc time in 49 pauses with 0 full sweep)
```
OK, that's enough fiddling around.  I hope this gave you enough to get your hands dirty.  We are already among the better results on the [MNIST website](http://yann.lecun.com/exdb/mnist) in the "permutation invariant, no distortion" category.  I am sure you can do better playing around with the learning rate, the momentum, adagrad and regularization, unit and layer types and counts etc.  But be careful, it could become addictive :)
