
<a id='Examples-1'></a>

# Examples


The following examples can be found in the [Knet/examples](https://github.com/denizyuret/Knet.jl/tree/master/examples) directory.


<a id='LinReg-1'></a>

## LinReg

<a id='LinReg' href='#LinReg'>#</a>
**`LinReg`** &mdash; *Module*.



LinReg is a simple linear regression example using artificially generated data. You can run the demo using `julia linreg.jl` on the command line or `julia> LinReg.main()` at the Julia prompt.  Use `julia linreg.jl --help` or `julia> LinReg.main("--help")` for a list of options.  The quadratic loss will be printed at every epoch and optimized parameters will be returned.


<a target='_blank' href='https://github.com/denizyuret/Knet.jl/tree/6123f3b740e5cc40925a23278f62fd83f2d70d10/examples/linreg.jl#L5-L14' class='documenter-source'>source</a><br>


<a id='Housing-1'></a>

## Housing

<a id='Housing' href='#Housing'>#</a>
**`Housing`** &mdash; *Module*.



This example uses the [Housing](https://archive.ics.uci.edu/ml/datasets/Housing) dataset from the UCI Machine Learning Repository to demonstrate a linear regression model. The dataset has housing related information for 506 neighborhoods in Boston from 1978. Each neighborhood has 14 attributes, the goal is to use the first 13, such as average number of rooms per house, or distance to employment centers, to predict the 14’th attribute: median dollar value of the houses.

You can run the demo using `julia housing.jl`.  Use `julia housing.jl --help` for a list of options.  The dataset will be automatically downloaded and randomly split into training and test sets.  The quadratic loss for the training and test sets will be printed at every epoch and optimized parameters will be returned.


<a target='_blank' href='https://github.com/denizyuret/Knet.jl/tree/6123f3b740e5cc40925a23278f62fd83f2d70d10/examples/housing.jl#L5-L21' class='documenter-source'>source</a><br>


<a id='MNIST-1'></a>

## MNIST

<a id='MNIST' href='#MNIST'>#</a>
**`MNIST`** &mdash; *Module*.



This example learns to classify hand-written digits from the [MNIST](http://yann.lecun.com/exdb/mnist) dataset.  There are 60000 training and 10000 test examples. Each input x consists of 784 pixels representing a 28x28 image.  The pixel values are normalized to [0,1]. Each output y is converted to a ten-dimensional one-hot vector (a vector that has a single non-zero component) indicating the correct class (0-9) for a given image.  10 is used to represent 0.

You can run the demo using `julia mnist.jl` on the command line or `julia> MNIST.main()` at the Julia prompt.  Options can be used like `julia mnist.jl --epochs 3` or `julia> MNIST.main("--epochs 3")`.  Use `julia mnist.jl --help` for a list of options.  The dataset will be automatically downloaded.  By default a softmax model will be trained for 10 epochs.  You can also train a multi-layer perceptron by specifying one or more –hidden sizes.  The accuracy for the training and test sets will be printed at every epoch and optimized parameters will be returned.


<a target='_blank' href='https://github.com/denizyuret/Knet.jl/tree/6123f3b740e5cc40925a23278f62fd83f2d70d10/examples/mnist.jl#L5-L25' class='documenter-source'>source</a><br>


<a id='LeNet-1'></a>

## LeNet

<a id='LeNet' href='#LeNet'>#</a>
**`LeNet`** &mdash; *Module*.



This example learns to classify hand-written digits from the [MNIST](http://yann.lecun.com/exdb/mnist) dataset.  There are 60000 training and 10000 test examples. Each input x consists of 784 pixels representing a 28x28 image.  The pixel values are normalized to [0,1]. Each output y is converted to a ten-dimensional one-hot vector (a vector that has a single non-zero component) indicating the correct class (0-9) for a given image.  10 is used to represent 0.

You can run the demo using `julia lenet.jl` at the command line or `julia> LeNet.main()` at the Julia prompt.  Use `julia lenet.jl --help` or `julia> LeNet.main("--help")` for a list of options.  The dataset will be automatically downloaded.  By default the [LeNet](http://yann.lecun.com/exdb/lenet) convolutional neural network model will be trained for 10 epochs.  The accuracy for the training and test sets will be printed at every epoch and optimized parameters will be returned.


<a target='_blank' href='https://github.com/denizyuret/Knet.jl/tree/6123f3b740e5cc40925a23278f62fd83f2d70d10/examples/lenet.jl#L7-L26' class='documenter-source'>source</a><br>


<a id='CharLM-1'></a>

## CharLM

<a id='CharLM' href='#CharLM'>#</a>
**`CharLM`** &mdash; *Module*.



This example implements an LSTM network for training and testing character-level language models inspired by ["The Unreasonable Effectiveness of Recurrent Neural Networks"](http://karpathy.github.io/2015/05/21/rnn-effectiveness) from Andrej Karpathy's blog.  The model can be trained with different genres of text, and can be used to generate original text in the same style.

Example usage:

  * `julia charlm.jl`: trains a model using its own code.
  * `julia charlm.jl --data foo.txt`: uses foo.txt to train instead.
  * `julia charlm.jl --data foo.txt bar.txt`: uses foo.txt for training and bar.txt for validation.  Any number of files can be specified, the first two will be used for training and validation, the rest for testing.
  * `julia charlm.jl --best foo.jld --save bar.jld`: saves the best model (according to validation set) to foo.jld, last model to bar.jld.
  * `julia charlm.jl --load foo.jld --generate 1000`: generates 1000 characters from the model in foo.jld.
  * `julia charlm.jl --help`: describes all available options.


<a target='_blank' href='https://github.com/denizyuret/Knet.jl/tree/6123f3b740e5cc40925a23278f62fd83f2d70d10/examples/charlm.jl#L12-L42' class='documenter-source'>source</a><br>


<a id='VGG-1'></a>

## VGG

<a id='VGG' href='#VGG'>#</a>
**`VGG`** &mdash; *Module*.



julia vgg.jl image-file-or-url

This example implements the VGG model from `Very Deep Convolutional Networks for Large-Scale Image Recognition', Karen Simonyan and Andrew Zisserman, arXiv technical report 1409.1556, 2014. This example works for D and E models currently. VGG-D is the default model if you do not specify any model.

  * Paper url: https://arxiv.org/abs/1409.1556
  * Project page: http://www.robots.ox.ac.uk/~vgg/research/very_deep
  * MatConvNet weights used here: http://www.vlfeat.org/matconvnet/pretrained


<a target='_blank' href='https://github.com/denizyuret/Knet.jl/tree/6123f3b740e5cc40925a23278f62fd83f2d70d10/examples/vgg.jl#L5-L19' class='documenter-source'>source</a><br>


<a id='ResNet-1'></a>

## ResNet

<a id='ResNet' href='#ResNet'>#</a>
**`ResNet`** &mdash; *Module*.



julia resnet.jl image-file-or-url

This example implements the ResNet-50, ResNet-101 and ResNet-152 models from 'Deep Residual Learning for Image Regocnition', Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, arXiv technical report 1512.03385, 2015.

  * Paper url: https://arxiv.org/abs/1512.03385
  * Project page: https://github.com/KaimingHe/deep-residual-networks
  * MatConvNet weights used here: http://www.vlfeat.org/matconvnet/pretrained


<a target='_blank' href='https://github.com/denizyuret/Knet.jl/tree/6123f3b740e5cc40925a23278f62fd83f2d70d10/examples/resnet.jl#L7-L19' class='documenter-source'>source</a><br>


<a id='Optimizers-1'></a>

## Optimizers

<a id='Optimizers' href='#Optimizers'>#</a>
**`Optimizers`** &mdash; *Module*.



This example demonstrates the usage of stochastic gradient descent(sgd) based optimization methods. We train LeNet model on MNIST dataset similar to `lenet.jl`.

You can run the demo using `julia optimizers.jl`.  Use `julia optimizers.jl --help` for a list of options. By default the [LeNet](http://yann.lecun.com/exdb/lenet) convolutional neural network model will be trained using sgd for 10 epochs. At the end of the training accuracy for the training and test sets for each epoch will be printed  and optimized parameters will be returned.


<a target='_blank' href='https://github.com/denizyuret/Knet.jl/tree/6123f3b740e5cc40925a23278f62fd83f2d70d10/examples/optimizers.jl#L8-L19' class='documenter-source'>source</a><br>


<a id='Overfitting,-underfitting,-regularization,-dropout-1'></a>

## Overfitting, underfitting, regularization, dropout


[softmax.ipynb](https://github.com/denizyuret/Knet.jl/tree/master/examples/softmax.ipynb) is an IJulia notebook demonstrating overfitting, underfitting, regularization, and dropout.

