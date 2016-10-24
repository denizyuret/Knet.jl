# Examples

## [LinReg](https://github.com/denizyuret/Knet.jl/blob/master/examples/linreg.jl)

LinReg is a simple linear regression example using artificially
generated data. You can run the demo using `julia linreg.jl`.  The
quadratic loss will be printed at every epoch and optimized parameters
will be returned.  Use `julia linreg.jl --help` for a list of options.

## [Housing](https://github.com/denizyuret/Knet.jl/blob/master/examples/housing.jl)

This example uses the
[Housing](https://archive.ics.uci.edu/ml/datasets/Housing) dataset
from the UCI Machine Learning Repository to demonstrate a linear
regression model. The dataset has housing related information for 506
neighborhoods in Boston from 1978. Each neighborhood has 14
attributes, the goal is to use the first 13, such as average number of
rooms per house, or distance to employment centers, to predict the
14’th attribute: median dollar value of the houses.

You can run the demo using `julia housing.jl`.  Use `julia housing.jl
--help` for a list of options.  The dataset will be automatically
downloaded and randomly split into training and test sets.  The
quadratic loss for the training and test sets will be printed at every
epoch and optimized parameters will be returned.

## [MNIST](https://github.com/denizyuret/Knet.jl/blob/master/examples/mnist.jl)

This example learns to classify hand-written digits from the
[MNIST](http://yann.lecun.com/exdb/mnist) dataset.  There are 60000
training and 10000 test examples. Each input x consists of 784 pixels
representing a 28x28 image.  The pixel values are normalized to
[0,1]. Each output y is converted to a ten-dimensional one-hot vector
(a vector that has a single non-zero component) indicating the correct
class (0-9) for a given image.  10 is used to represent 0.

You can run the demo using `julia mnist.jl`.  Use `julia mnist.jl
--help` for a list of options.  The dataset will be automatically
downloaded.  By default a softmax model will be trained for 10 epochs.
You can also train a multi-layer perceptron by specifying one or more
--hidden sizes.  The accuracy for the training and test sets will be
printed at every epoch and optimized parameters will be returned.

## [LeNet](https://github.com/denizyuret/Knet.jl/blob/master/examples/lenet.jl)

This example learns to classify hand-written digits using the same
dataset as the MNIST demo and the
[LeNet](http://yann.lecun.com/exdb/lenet) convolutional neural network
model.  Note that you need a GPU machine for this demo.  You can run
the demo using `julia lenet.jl`.  The dataset will be automatically
downloaded.  The accuracy for the training and test sets will be
printed at every epoch and optimized parameters will be returned.  Use
`julia lenet.jl --help` for a list of options.

## [CharLM](https://github.com/denizyuret/Knet.jl/blob/master/examples/charlm.jl)

This example implements an LSTM network for training and testing
character-level language models inspired by ["The Unreasonable
Effectiveness of Recurrent Neural
Networks"](http://karpathy.github.io/2015/05/21/rnn-effectiveness)
from the Andrej Karpathy blog.  The model can be trained with
different genres of text, and can be used to generate original text in
the same style.  You can run the demo with default settings using
`julia charlm.jl`.  This will download and train a model on "The
Complete Works of William Shakespeare".  Note that this will take
considerably longer than the other demos.  Please see the
documentation in the code or use `julia charlm.jl --help` for a list
of options.

## [VGG](https://github.com/denizyuret/Knet.jl/blob/master/examples/vgg.jl)

This example implements the [VGG
model](http://www.robots.ox.ac.uk/~vgg/research/very_deep) from ["Very
Deep Convolutional Networks for Large-Scale Image
Recognition"](https://arxiv.org/abs/1409.1556), by Karen Simonyan and
Andrew Zisserman, arXiv technical report 1409.1556, 2014.  In
particular we use the 16 layer network, denoted as configuration D in
the technical report.  The pretrained weights will be downloaded from
the [MatConvNet website](http://www.vlfeat.org/matconvnet/pretrained)
(492MB) the first time the program is used.  You can then classify an
image from a local file or URL using `julia vgg.jl image-file-or-url`.
