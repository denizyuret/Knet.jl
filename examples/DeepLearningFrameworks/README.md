# DeepLearningFrameworks (last updated Dec 7, 2017)

- [Knet_CNN.ipynb](Knet_CNN.ipynb): Convolutional neural network trained on [CIFAR-10].
- [Knet_RNN.ipynb](Knet_RNN.ipynb): RNN trained on [IMDB] movie reviews sentiment classification.
- [ResNet50-Knet.ipynb](ResNet50-Knet.ipynb): Inference (feature extraction) with [ResNet-50].

This directory contains the Knet notebooks submitted to
[DeepLearningFrameworks]. Many thanks to [Ilia
Karmanov](https://github.com/ilkarman) for creating and maintaining
this benchmark. The same models have been implemented with CNTK,
Caffe2, Chainer, Gluon, Keras, Knet, Lasagne, MXNet, PyTorch and
TensorFlow to give both coding style and speed comparisons.  Here are
the results with links to the original sources and hardware specs:

[DeepLearningFrameworks]: https://github.com/ilkarman/DeepLearningFrameworks
[IMDB]: http://ai.stanford.edu/~amaas/data/sentiment
[CIFAR-10]: http://www.cs.toronto.edu/~kriz/cifar.html
[ResNet-50]: https://github.com/KaimingHe/deep-residual-networks

## Training CNN (VGG-style) on CIFAR-10 - Image Recognition

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

## Training RNN (GRU) on IMDB - Natural Language Processing (Sentiment Analysis)

| DL Library                          | Test Accuracy (%) | Training Time (s) | Using CuDNN? |
| ----------------------------------- | ----------------- | ----------------- | ------------ |
| [MXNet](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/MXNet_RNN.ipynb)            | 86                | 29                | Yes          |
| [Knet(Julia)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Knet_RNN.ipynb)       | 85                | 29                | Yes          |
| [Tensorflow](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Tensorflow_RNN.ipynb)  | 86                | 30                | Yes          |
| [Pytorch](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/PyTorch_RNN.ipynb)        | 86                | 31                | Yes          |
| [CNTK](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/CNTK_RNN.ipynb)              | 85                | 32                | Yes          |
| [Keras(TF)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Keras_TF_RNN.ipynb)     | 86                | 35                | Yes          |
| [Keras(CNTK)](https://github.com/ilkarman/DeepLearningFrameworks/blob/master/Keras_CNTK_RNN.ipynb) | 86                | 86                | No Available |

## Inference ResNet-50 (Feature Extraction)

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

## Hardware specs

The benchmarks were run on (half) an Nvidia K80 GPU, on [Microsoft
Azure Deep Learning Virtual
Machine](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning?tab=Overview),
[NC6](https://azure.microsoft.com/en-gb/blog/azure-n-series-preview-availability/),
where frameworks have been updated to the latest version.

| Type   | Spec |
|:-------|:-----|
| Cores  | 6 x E5-2690v3 |
| GPU    | 1 x K80 (1/2 Physical Card) |
| Memory | 56 GB |
| Disk   | 380 GB SSD |

