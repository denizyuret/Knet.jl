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
