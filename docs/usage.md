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
parameters like the learningRate can be specified during layer
construction, as well as afterward using setparam!.  It is also
possible to save layers to HDF5 files using `h5write(fname::String,
l::Layer)` and read them using `Layer(fname::String)`.  Please see
`types.jl` and `h5io.jl` for details.

