## Layers

A layer represents a primitive operation (e.g. matrix multiplication,
activation function) in KUnet.  One design decision in KUnet has been
to define layers as fine grained as possible (e.g. separate mmul,
bias, relu into their own layers) to reduce the number of
configuration options an facilitate code reuse.  Here is a list of
layers implemented:

* [Mmul](https://github.com/denizyuret/KUnet.jl/blob/master/src/mmul.jl), [Bias](https://github.com/denizyuret/KUnet.jl/blob/master/src/bias.jl) matrix multiplication and bias for feed forward nets.
* [Conv](https://github.com/denizyuret/KUnet.jl/blob/master/src/conv.jl), [Pool](https://github.com/denizyuret/KUnet.jl/blob/master/src/pool.jl) convolution and pooling for convolutional nets.
* [Add2](https://github.com/denizyuret/KUnet.jl/blob/master/src/add2.jl), [Mul2](https://github.com/denizyuret/KUnet.jl/blob/master/src/mul2.jl) elementwise addition and multiplication for recurrent nets.
* [Activation Functions](actf.md) e.g. sigmoid, tanh and relu.
* [Loss Layers](loss.md) e.g. cross entropy, quadratic.
* [Perceptrons](perceptron.md) perceptrons and kernel perceptrons.
* [Drop](https://github.com/denizyuret/KUnet.jl/blob/master/src/drop.jl) dropout.

Feed forward, convolutional, recurrent nets and perceptrons are
constructed by gluing together layers.  For the glue to work, each
layer has to follow a common interface.  For efficiency, parts of this
interface has to be flexible (e.g. some layers allocate their outputs,
others overwrite their inputs to minimize memory usage).  We specify
the defaults and the exceptions below.

