# Examples

- [julia-tutorial](julia-tutorial): Julia examples demonstrating arrays, tuples, dictionaries, indexing etc.
- [knet-tutorial](knet-tutorial): Notebook with Knet models for linreg, softmax, mlp, cnn, rnn.
- [synthetic-linreg](synthetic-linreg): Simple linear regression example using artificial data.
- [housing-linreg](housing-linreg): Linear regression on the [Boston Housing] dataset.
- [mnist-mlp](mnist-mlp): Multi-layer perceptron trained on [MNIST].
- [fashion-mnist](fashion-mnist): Multi-layer perceptron trained on [Fashion-MNIST].
- [cifar10-cnn](cifar10-cnn): Notebook with convolutional neural network trained on [CIFAR-10].
- [lenet](lenet): The [LeNet] model trained on [MNIST].
- [rnn-tutorial](rnn-tutorial): RNN tutorial notebook with BPTT, LSTM, S2S.
- [imdb-rnn](imdb-rnn): Notebook with RNN trained on [IMDB] movie reviews sentiment classification.
- [charlm](charlm): Character-level RNN language model from [Karpathy].
- [rnnlm](rnnlm): Word-level RNN language model trained on the [Mikolov-PTB] corpus.
- [optimizers](optimizers): Try various optimizers (SGD, Momentum, Adam...) on [LeNet].
- [overfitting](overfitting): Notebook on underfitting, overfitting, regularization, dropout.
- [resnet](resnet): Knet implementation of [ResNet] 50, 101, and 152 models.
- [vgg](vgg): Knet implementation of [VGG] D and E models.
- [bilstm-taggers](bilstm-taggers) Two different BiLSTM taggers on [WikiNER] data. [One](bilstm-taggers/bilstm-tagger.jl) has the word UNK in its vocab, [the other one](bilstm-taggers/bilstm-tagger-withchar.jl) generates UNK words' embeddings by using an another BiLSTM network which takes characters as input.
- [variational-autoencoder](variational-autoencoder): Train a Variational Autoencoder on [MNIST].

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
[WIKINER]: https://github.com/neulab/dynet-benchmark/tree/master/data/tags
