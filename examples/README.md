# Examples

## Tutorials
- [julia-tutorial](julia-tutorial): Julia examples demonstrating arrays, tuples, dictionaries, indexing etc.
- [knet-tutorial](knet-tutorial): Notebook with Knet models for linreg, softmax, mlp, cnn, rnn.
- [optimizers](optimizers): Try various optimizers (SGD, Momentum, Adam...) on [LeNet].
- [overfitting](overfitting): Notebook on underfitting, overfitting, regularization, dropout.
- [rnn-tutorial](rnn-tutorial): RNN tutorial notebook with BPTT, LSTM, S2S.

## Benchmarks
- [DeepLearningFrameworks](DeepLearningFrameworks): Notebooks comparing CNTK, Caffe2, Chainer, Gluon, Keras, Knet, Lasagne, MXNet, PyTorch, TensorFlow on CNN and RNN examples.
- [dynet-benchmark](dynet-benchmark): Four dynamic neural network examples comparing Knet with DyNet and Chainer from [dynet-benchmark](https://github.com/neulab/dynet-benchmark).

## Models

### Linear
- [synthetic-linreg](synthetic-linreg): Simple linear regression example using artificial data.
- [housing-linreg](housing-linreg): Linear regression on the [Boston Housing] dataset.

### MLP
- [mnist-mlp](mnist-mlp): Multi-layer perceptron trained on [MNIST].
- [fashion-mnist](fashion-mnist): Multi-layer perceptron trained on [Fashion-MNIST].

### CNN
- [lenet](lenet): The [LeNet] model trained on [MNIST].
- [cifar10-cnn](cifar10-cnn): CNN model for [CIFAR-10] with batchnorm.
- [resnet](resnet): Knet implementation of [ResNet] 50, 101, and 152 models.
- [vgg](vgg): Knet implementation of [VGG] D and E models.

### RNN
- [charlm](charlm): Character-level RNN language model from [Karpathy].
- [rnnlm](rnnlm): Word-level RNN language model trained on the [Mikolov-PTB] corpus.

### Other
- [variational-autoencoder](variational-autoencoder): Train a Variational Autoencoder on [MNIST].
- [dcgan-mnist](dcgan-mnist): Train a [DCGAN](https://arxiv.org/abs/1511.06434) (Deep Convolutional Generative Adversarial Network) on [MNIST].


[MNIST]: http://yann.lecun.com/exdb/mnist
[LeNet]: http://yann.lecun.com/exdb/lenet
[Boston Housing]: https://archive.ics.uci.edu/ml/machine-learning-databases/housing
[CIFAR-10]: http://www.cs.toronto.edu/~kriz/cifar.html
[IMDB]: http://ai.stanford.edu/~amaas/data/sentiment
[Shakespeare]: http://www.gutenberg.org/ebooks/100
[Mikolov-PTB]: http://www.fit.vutbr.cz/~imikolov/rnnlm
[Fashion-MNIST]: https://github.com/zalandoresearch/fashion-mnist
[Karpathy]: http://karpathy.github.io/2015/05/21/rnn-effectiveness
[ResNet]: https://github.com/KaimingHe/deep-residual-networks
[VGG]: http://www.robots.ox.ac.uk/~vgg/research/very_deep
