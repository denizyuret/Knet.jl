# Introduction to Knet

## Summary

[Knet](https://denizyuret.github.io/Knet.jl/latest) (pronounced "kay-net") is the [Koç
University](http://www.ku.edu.tr/en) deep learning framework implemented in
[Julia](http://docs.julialang.org) by [Deniz Yuret](http://www.denizyuret.com) and
collaborators.  It supports GPU operation and automatic differentiation using dynamic
computational graphs for models defined in plain Julia. You can install Knet with the
following at the julia prompt: `using Pkg; Pkg.add("Knet")`. Some useful links:

* [Tutorial:](https://github.com/denizyuret/Knet.jl/tree/master/tutorial) 
  introduces Julia and Knet via examples.
* [Documentation:](https://denizyuret.github.io/Knet.jl/latest)
  installation, introduction, design, implementation, full reference and deep learning chapters.
* [Examples:](https://github.com/denizyuret/Knet.jl/tree/master/examples)
  more tutorials and example models.
* [Benchmarks:](http://denizyuret.github.io/Knet.jl/latest/tutorial.html#Benchmarks-1)
  comparison of Knet's speed with TensorFlow, PyTorch, DyNet etc.
* [Paper:](https://goo.gl/zeUBFr)
  Yuret, D. "Knet: beginning deep learning with 100 lines of julia." In *Machine Learning Systems Workshop* at NIPS 2016.
* [KnetML:](https://github.com/KnetML)
  github organization with Knet repos of models, tutorials, layer collections and other resources.
* [Images:](http://denizyuret.github.io/Knet.jl/latest/install.html#Using-Amazon-AWS-1)
  Knet machine images are available for [AWS](http://denizyuret.github.io/Knet.jl/latest/install.html#Using-Amazon-AWS-1), [Singularity](https://github.com/KnetML/singularity-images) and [Docker](https://github.com/JuliaGPU/docker).
* [Issues:](https://github.com/denizyuret/Knet.jl/issues)
  if you find a bug, please open a github issue.
* [knet-users:](https://groups.google.com/forum/#!forum/knet-users)
  if you need help or would like to request a feature, please join this mailing list.
* [knet-dev:](https://groups.google.com/forum/#!forum/knet-dev)
  if you would like to contribute to Knet development, please join this mailing list and check out these [tips](http://denizyuret.github.io/Knet.jl/latest/install.html#Tips-for-developers-1).
* [knet-slack:](https://julialang.slack.com/messages/CDLKQ92P3/details) Slack channel for Knet.
* Related work: Please check out [Flux](https://github.com/FLuxML), [Mocha](https://github.com/pluskid/Mocha.jl), [JuliaML](https://github.com/JuliaML), [JuliaDiff](https://github.com/JuliaDiff), [JuliaGPU](https://github.com/JuliaGPU), [JuliaOpt](https://github.com/JuliaOpt) for related packages.

## Philosophy

Knet uses dynamic computational graphs generated at runtime for automatic differentiation of
(almost) any Julia code.  This allows machine learning models to be implemented by defining
just the forward calculation (i.e. the computation from parameters and data to loss) using
the full power and expressivity of Julia. The implementation can use helper functions,
loops, conditionals, recursion, closures, tuples and dictionaries, array indexing,
concatenation and other high level language features, some of which are often missing in the
restricted modeling languages of static computational graph systems like Theano, Torch,
Caffe and Tensorflow.  GPU operation is supported by simply using the KnetArray type instead
of regular Array for parameters and data.

Knet builds a dynamic computational graph by recording primitive operations during forward
calculation.  Only pointers to inputs and outputs are recorded for efficiency.  Therefore
array overwriting is not supported during forward and backward passes.  This encourages a
clean functional programming style.  High performance is achieved using custom memory
management and efficient GPU kernels.  See [Under the hood](@ref) for more details.


## Tutorial

The Knet tutorial consists of Jupyter notebooks that introduce the programming language
Julia and the Knet deep learning framework. By the end, the reader should be able to define,
train, evaluate, and visualize basic MLP, CNN, and RNN models.  Each notebook is written to
work stand-alone but they rely on concepts introduced in earlier notebooks, so I recommend
reading them in order. Every Knet function outside of the standard Julia library is defined
or explained before use. You can view the notebooks using the following links, or interact
with them using a Jupyter server. Instructions for running a server locally or in the cloud
can be found in the tutorial
[README](https://github.com/denizyuret/Knet.jl/tree/master/tutorial/README.md).

* [Julia is fast:](https://github.com/denizyuret/Knet.jl/blob/master/tutorial/00.Julia_is_fast.ipynb)
  comparison of Julia's speed to C, Python and numpy.
* [Getting to know Julia:](https://github.com/denizyuret/Knet.jl/blob/master/tutorial/10.Getting_to_know_Julia.ipynb)
  basic Julia tutorial from [JuliaBox](http://juliabox.com).
* [Quick start:](https://github.com/denizyuret/Knet.jl/blob/master/tutorial/15.quickstart.ipynb)
  if you are familiar with other deep learning frameworks and want to see a quick Julia example.
* [The MNIST dataset:](https://github.com/denizyuret/Knet.jl/blob/master/tutorial/20.mnist.ipynb)
  introduction to the MNIST handwritten digit recognition dataset.
* [Julia iterators:](https://github.com/denizyuret/Knet.jl/blob/master/tutorial/25.iterators.ipynb)
  iterators are useful for generating and training with data.
* [Creating a model:](https://github.com/denizyuret/Knet.jl/blob/master/tutorial/30.lin.ipynb)
  define, train, visualize simple linear models, introduce gradients, SGD, using the GPU.
* [Multilayer perceptrons:](https://github.com/denizyuret/Knet.jl/blob/master/tutorial/40.mlp.ipynb)
  multi layer perceptrons, nonlinearities, model capacity, overfitting, regularization, dropout.
* [Convolutional networks:](https://github.com/denizyuret/Knet.jl/blob/master/tutorial/50.cnn.ipynb)
  convolutional neural networks, sparse and shared weights using conv4 and pool operations.
* [Recurrent networks:](https://github.com/denizyuret/Knet.jl/blob/master/tutorial/60.rnn.ipynb)
  introduction to recurrent neural networks.
* [IMDB sentiment analysis:](https://github.com/denizyuret/Knet.jl/blob/master/tutorial/70.imdb.ipynb)
  a simple RNN sequence classification model for sentiment analysis of IMDB movie reviews.
* [Language modeling:](https://github.com/denizyuret/Knet.jl/blob/master/tutorial/80.charlm.ipynb)
  a character based RNN language model that can write Shakespeare sonnets and Julia programs.
* [Sequence to sequence:](https://github.com/denizyuret/Knet.jl/blob/master/tutorial/90.s2s.ipynb)
  a sequence to sequence RNN model typically used for machine translation.


## Benchmarks

### Knet Benchmarks (Sep 30, 2016)

Each of the examples above was used as a benchmark to compare Knet with other
frameworks. The table below shows the number of seconds it takes to train a given model for
a particular dataset, number of epochs and minibatch size for Knet, Theano, Torch, Caffe and
TensorFlow. Knet had comparable performance to other commonly used frameworks.

|model|dataset|epochs|batch|Knet|Theano|Torch|Caffe|TFlow|
|:----|:------|:-----|:----|:---|:-----|:----|:----|:----|
|LinReg|Housing|10K|506|2.84|1.88|2.66|2.35|5.92|
|Softmax|MNIST|10|100|2.35|1.40|2.88|2.45|5.57|
|MLP|MNIST|10|100|3.68|2.31|4.03|3.69|6.94|
|LeNet|MNIST|1|100|3.59|3.03|1.69|3.54|8.77|
|CharLM|Hiawatha|1|128|2.25|2.42|2.23|1.43|2.86|

The benchmarking was done on g2.2xlarge GPU instances on Amazon AWS. The code is available
at [github](https://github.com/ozanarkancan/Knet8-Benchmarks) and as machine image
`deep_AMI_v6` at AWS N.California. See the section on [Using Amazon AWS](@ref) for more
information. The datasets are available online using the following links:
[Housing](https://archive.ics.uci.edu/ml/datasets/Housing),
[MNIST](http://yann.lecun.com/exdb/mnist),
[Hiawatha](http://www.gutenberg.org/files/19/19.txt). The MLP uses a single hidden layer of
64 units. CharLM uses a single layer LSTM language model with embedding and hidden layer
sizes set to 256 and trained using BPTT with a sequence length of 100. Each dataset was
minibatched and transferred to GPU prior to benchmarking when possible.


### DyNet Benchmarks (Dec 15, 2017)

We implemented dynamic neural network examples from the
[dynet-benchmark](https://github.com/neulab/dynet-benchmark) repo to compare Knet with DyNet
and Chainer. See [DyNet technical report](https://arxiv.org/abs/1701.03980) for the
architectural details of the implemented examples and the [github
repo](https://github.com/neulab/dynet-benchmark) for the source code.

- [rnnlm-batch](https://github.com/denizyuret/Knet.jl/blob/master/examples/dynet-benchmark/rnnlm-batch.jl): A recurrent neural network language model on [PTB](https://catalog.ldc.upenn.edu/ldc99t42) corpus.
- [bilstm-tagger](https://github.com/denizyuret/Knet.jl/blob/master/examples/dynet-benchmark/bilstm-tagger.jl): A bidirectional LSTM network that predicts a tag for each word. It is trained on [WikiNER](https://github.com/neulab/dynet-benchmark/tree/master/data/tags) dataset.
- [bilstm-tagger-withchar](https://github.com/denizyuret/Knet.jl/blob/master/examples/dynet-benchmark/bilstm-tagger-withchar.jl): Similar to bilstm-tagger, but uses characer-based embeddings for unknown words.
- [treenn](https://github.com/denizyuret/Knet.jl/blob/master/examples/dynet-benchmark/treenn.jl): A tree-structured LSTM sentiment classifier trained on [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html) dataset.

Benchmarks were run on a server with Intel(R) Xeon(R) CPU E5-2695 v4 @
2.10GHz and Tesla K80.

| Model                                               | Metric    |  Knet    | DyNet     | Chainer     |
| ----------------------------------------------------| --------- | -------- | --------- |------------ |
| [rnnlm-batch](https://github.com/denizyuret/Knet.jl/blob/master/examples/dynet-benchmark/rnnlm-batch.jl)                       | words/sec | 28.5k    | 18.7k     | 16k         |
| [bilstm-tagger](https://github.com/denizyuret/Knet.jl/blob/master/examples/dynet-benchmark/bilstm-tagger.jl)                   | words/sec | 6800     | 1200      | 157         |
| [bilstm-tagger-withchar](https://github.com/denizyuret/Knet.jl/blob/master/examples/dynet-benchmark/bilstm-tagger-withchar.jl) | words/sec | 1300     | 900       | 128         |
| [treenn](https://github.com/denizyuret/Knet.jl/blob/master/examples/dynet-benchmark/treenn.jl)                                 | sents/sec | 43       | 68        | 10          |


### DeepLearningFrameworks (Nov 24, 2017)

More recently, @ilkarman has published CNN and RNN
[benchmarks](https://github.com/ilkarman/DeepLearningFrameworks) on Nvidia K80 GPUs, using
the Microsoft Azure Data Science Virtual Machine for Linux (Ubuntu). The results are copied
below.  You can find versions of the Knet notebooks used for these benchmarks in the
Knet/examples/DeepLearningFrameworks directory.

Training CNN (VGG-style) on CIFAR-10 - Image Recognition

| DL Library                               | Test Accuracy (%) | Training Time (s) |
| ---------------------------------------- | ----------------- | ----------------- |
| [MXNet](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/MXNet_CNN.ipynb)                 | 77                | 145               |
| [Caffe2](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Caffe2_CNN.ipynb)               | 79                | 148               |
| [Gluon](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Gluon_CNN.ipynb)                 | 76                | 152               |
| [Knet(Julia)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Knet_CNN.ipynb)            | 78                | 159               |
| [Chainer](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Chainer_CNN.ipynb)             | 79                | 162               |
| [CNTK](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/CNTK_CNN.ipynb)                   | 78                | 163               |
| [PyTorch](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/PyTorch_CNN.ipynb)             | 78                | 169               |
| [Tensorflow](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Tensorflow_CNN.ipynb)       | 78                | 173               |
| [Keras(CNTK)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Keras_CNTK_CNN.ipynb)      | 77                | 194               |
| [Keras(TF)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Keras_TF_CNN.ipynb)          | 77                | 241               |
| [Lasagne(Theano)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Theano_Lasagne_CNN.ipynb) | 77                | 253               |
| [Keras(Theano)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Keras_Theano_CNN.ipynb)  | 78                | 269               |

Training RNN (GRU) on IMDB - Natural Language Processing (Sentiment Analysis)

| DL Library                          | Test Accuracy (%) | Training Time (s) | Using CuDNN? |
| ----------------------------------- | ----------------- | ----------------- | ------------ |
| [MXNet](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/MXNet_RNN.ipynb)            | 86                | 29                | Yes          |
| [Knet(Julia)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Knet_RNN.ipynb)       | 85                | 29                | Yes          |
| [Tensorflow](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Tensorflow_RNN.ipynb)  | 86                | 30                | Yes          |
| [Pytorch](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/PyTorch_RNN.ipynb)        | 86                | 31                | Yes          |
| [CNTK](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/CNTK_RNN.ipynb)              | 85                | 32                | Yes          |
| [Keras(TF)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Keras_TF_RNN.ipynb)     | 86                | 35                | Yes          |
| [Keras(CNTK)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Keras_CNTK_RNN.ipynb) | 86                | 86                | N/A          |

Inference ResNet-50 (Feature Extraction)

| DL Library                                          | Images/s GPU      | Images/s CPU      |
| ----------------------------------------            | ----------------- | ----------------- |
| [Knet(Julia)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/inference/ResNet50-Knet.ipynb)        | 160               | 2                 |
| [Tensorflow](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/inference/ResNet50-TF.ipynb)           | 155               | 11                |
| [PyTorch](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/inference/ResNet50-PyTorch.ipynb)         | 130               | 6                 |
| [MXNet](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/inference/ResNet50-MXNet.ipynb)             | 130               | 8                 |
| [MXNet(w/mkl)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/inference/ResNet50-MXNet-mkl.ipynb)  | 129               | 25                |
| [CNTK](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/inference/ResNet50-CNTK.ipynb)               | 117               | 8                 |
| [Chainer](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/inference/ResNet50-Chainer.ipynb)         | 107               | 3                 |
| [Keras(TF)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/inference/ResNet50-Keras(TF).ipynb)     | 98                | 5                 |
| [Caffe2](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/inference/ResNet50-Caffe2.ipynb)           | 71                | 6                 |
| [Keras(CNTK)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/inference/ResNet50-Keras(CNTK).ipynb) | 46                | 4                 |


## Under the hood

Knet relies on the [AutoGrad](https://github.com/denizyuret/AutoGrad.jl) package and the
[KnetArray](@ref) data type for its functionality and performance. AutoGrad computes the
gradient of Julia functions and KnetArray implements high performance GPU arrays with custom
memory management. This section briefly describes them.

### KnetArrays

GPUs have become indispensable for training large deep learning models.  Even the small
examples implemented here run up to 17x faster on the GPU compared to the 8 core CPU
architecture we use for benchmarking. However GPU implementations have a few potential
pitfalls: (i) GPU memory allocation is slow, (ii) GPU-RAM memory transfer is slow, (iii)
reduction operations (like `sum`) can be very slow unless implemented properly (See
[Optimizing Parallel Reduction in
CUDA](http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf)).

Knet implements [KnetArray](@ref) as a Julia data type that wraps GPU array
pointers. KnetArray is based on the more standard
[CudaArray](https://github.com/JuliaGPU/CUDArt.jl) with a few important differences: (i)
KnetArrays have a custom memory manager, similar to [ArrayFire](http://arrayfire.com), which
reuse pointers garbage collected by Julia to reduce the number of GPU memory allocations,
(ii) contiguous array ranges (e.g. `a[:,3:5]`) are handled as views with shared pointers
instead of copies when possible, and (iii) a number of custom CUDA kernels written for
KnetArrays implement element-wise, broadcasting, and scalar and vector reduction operations
efficiently. As a result Knet allows users to implement their models using high-level code,
yet be competitive in performance with other frameworks as demonstrated in the benchmarks
section. Other GPU related Julia packages can be found in [JuliaGPU](https://github.com/JuliaGPU).

### AutoGrad

As we have seen, many common machine learning models can be expressed as differentiable
programs that input parameters and data and output a scalar loss value. The loss value
measures how close the model predictions are to desired values with the given
parameters. Training a model can then be seen as an optimization problem: find the
parameters that minimize the loss. Typically, a gradient based optimization algorithm is
used for computational efficiency: the direction in the parameter space in which the loss
reduction is maximum is given by the negative gradient of the loss with respect to the
parameters. Thus gradient computations take a central stage in software frameworks for
machine learning. In this section I will briefly outline existing gradient computation
techniques and motivate the particular approach taken by Knet.

Computation of gradients in computer models is performed by four main methods [(Baydin et
al. 2015)](https://arxiv.org/abs/1502.05767):

-   manual differentiation (programming the derivatives)
-   numerical differentiation (using finite difference approximations)
-   symbolic differentiation (using expression manipulation)
-   automatic differentiation (detailed below)

Manually taking derivatives and coding the result is labor intensive, error-prone, and all
but impossible with complex deep learning models.  Numerical differentiation is simple:
$f'(x)=(f(x+\epsilon)-f(x-\epsilon))/(2\epsilon)$ but impractical: the finite difference
equation needs to be evaluated for each individual parameter, of which there are typically
many. Pure symbolic differentiation using expression manipulation, as implemented in
software such as Maxima, Maple, and Mathematica is impractical for different reasons: (i) it
may not be feasible to express a machine learning model as a closed form mathematical
expression, and (ii) the symbolic derivative can be exponentially larger than the model
itself leading to inefficient run-time calculation. This leaves us with automatic
differentiation.

Automatic differentiation is the idea of using symbolic derivatives only at the level of
elementary operations, and computing the gradient of a compound function by applying the
chain rule to intermediate numerical results. For example, pure symbolic differentiation of
$\sin^2(x)$ could give us $2\sin(x)\cos(x)$ directly. Automatic differentiation would use
the intermediate numerical values $x_1=\sin(x)$, $x_2=x_1^2$ and the elementary derivatives
$dx_2/dx_1=2x_1$, $dx_1/dx=\cos(x)$ to compute the same answer without ever building a full
gradient expression.

To implement automatic differentiation the target function needs to be decomposed into its
elementary operations, a process similar to compilation. Most older machine learning
frameworks (such as Theano, Torch, Caffe, Tensorflow and older versions of Knet prior to
v0.8) compile models expressed in a restricted mini-language into a static computational
graph of elementary operations that have pre-defined derivatives. There are two drawbacks
with this approach: (i) the restricted mini-languages tend to have limited support for
high-level language features such as conditionals, loops, helper functions, array indexing,
etc. (e.g. the infamous `scan` operation in Theano) (ii) the sequence of elementary
operations that unfold at run-time needs to be known in advance, and they are difficult to
handle when the sequence is data dependent.

There is an alternative: high-level languages, like Julia and Python, already know how to
decompose functions into their elementary operations. If we let the users define their
models directly in a high-level language, then record the elementary operations during loss
calculation at run-time, a dynamic computational graph can be constructed from the recorded
operations. The cost of recording is not prohibitive: The table below gives cumulative times
for elementary operations of an MLP with quadratic loss. Recording only adds 15% to the raw
cost of the forward computation. Backpropagation roughly doubles the total time as expected.

|op|secs|
|:--|:---|
|`a1=w1*x`|0.67|
|`a2=w2.+a1`|0.71|
|`a3=max.(0,a2)`|0.75|
|`a4=w3*a3`|0.81|
|`a5=w4.+a4`|0.85|
|`a6=a5-y`|0.89|
|`a7=sum(abs2,a6)`|1.18|
|+recording|1.33|
|+backprop|2.79|


This is the approach taken by the popular [autograd](https://github.com/HIPS/autograd)
Python package and its Julia port [AutoGrad.jl](https://github.com/denizyuret/AutoGrad.jl)
used by Knet. Recently, other machine learning frameworks have been adapting dynamic
computational graphs: [Chainer](http://docs.chainer.org/en/stable/index.html),
[DyNet](https://arxiv.org/abs/1701.03980), [PyTorch](https://github.com/pytorch/pytorch),
[TensorFlow
Fold](https://research.googleblog.com/2017/02/announcing-tensorflow-fold-deep.html). Related
Julia projects include [Flux](https://github.com/FLuxML) and
[JuliaDiff](https://github.com/JuliaDiff).

In AutoGrad, parameters of interest are boxed by the `Param` type. `y = @diff f(x)` returns
a struct such that `value(y)` gives `f(x)` (which should be a scalar), `params(y)` gives the
list of parameters that took place in the computation of `f(x)`, and `grad(y,p)` gives the
gradient of `f(x)` with respect to parameter `p`.  In a `@diff` context, the elementary
operations in `f` are overloaded to record their actions and output boxed answers when their
inputs are boxed. The sequence of recorded operations is then used to compute
gradients. Derivatives can be defined independently for each method of a function
(determined by argument types) making full use of Julia's multiple dispatch. New elementary
operations and derivatives can be defined concisely using Julia's macro and meta-programming
facilities. See [AutoGrad.jl](https://github.com/denizyuret/AutoGrad.jl) for details.
