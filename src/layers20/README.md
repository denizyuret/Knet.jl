# Knet.Layers20

Knet.Layers20 is a submodule that provides useful deep learning layers for [Knet](https://github.com/denizyuret/Knet.jl), fostering your model development. It was originally developed as the independent KnetLayers package by @ekinakyurek.

## Overview
```JULIA
model = Chain(Dense(input=768, output=128, activation=Sigm()),
	      Dense(input=128, output=10, activation=nothing))

loss(model, x, y) = nll(model(x), y)
```

## Getting Started: Train an MNIST model
```Julia
using Knet, Knet.Layers20
import Knet: Data
#Data
include(Knet.dir("data","mnist.jl"))
dtrn,dtst = mnistdata(xsize=(784,:)); # dtrn and dtst = [ (x1,y1), (x2,y2), ... ] where xi,yi are

#Model
HIDDEN_SIZES = [100,50]
(m::MLP)(x,y) = nll(m(x),y)
(m::MLP)(d::Data) = mean(m(x,y) for (x,y) in d)
model = MLP(784,HIDDEN_SIZES...,10)

#Train
EPOCH=10
progress!(sgd(model,repeat(dtrn,EPOCH)))

#Test
@show 100accuracy(model, dtst)
```

## Example Models

1) [MNIST-MLP](./examples/mnist.jl)

2) [MNIST-CNN](./examples/mnist-cnn.jl)

3) [GAN-MLP](./examples/gan-mlp.ipynb)

4) [ResNet: Residual Networks for Image Recognition](./examples/resnet.jl)

5) [S2S: Sequence to Sequence Reccurent Model](./examples/s2smodel.jl)

6) [Morse.jl: Morphological Analyzer+Lemmatizer](https://github.com/ekinakyurek/Morse.jl)

7) [MAC Network: Memory-Attention-Composition Network for Visual Question Answering](https://github.com/ekinakyurek/Mac-Network)

## [Exported Layers Refence](https://ekinakyurek.github.io/KnetLayers.jl/latest/reference.html#Function-Index-1)

## Example Layers and Usage
```JULIA
using Knet.Layers20

#Instantiate an MLP model with random parameters
mlp = MLP(100,50,20; activation=Sigm()) # input size=100, hidden=50 and output=20

#Do a prediction with the mlp model
prediction = mlp(randn(Float32,100,1))

#Instantiate a convolutional layer with random parameters
cnn = Conv(height=3, width=3, inout=3=>10, padding=1, stride=1) # A conv layer

#Filter your input with the convolutional layer
output = cnn(randn(Float32,224,224,3,1))

#Instantiate an LSTM model
lstm = LSTM(input=100, hidden=100, embed=50)

#You can use integers to represent one-hot vectors.
#Each integer corresponds to vocabulary index of corresponding element in your data.

#For example a pass over 5-Length sequence
rnnoutput = lstm([3,2,1,4,5];hy=true,cy=true)

#After you get the output, you may acces to hidden states and
#intermediate hidden states produced by the lstm model
rnnoutput.y
rnnoutput.hidden
rnnoutput.memory

#You can also use normal array inputs for low-level control
#One iteration of LSTM with a random input
rnnoutput = lstm(randn(100,1);hy=true,cy=true)

#Pass over a random 10-length sequence:
rnnoutput = lstm(randn(100,1,10);hy=true,cy=true)

#Pass over a mini-batch data which includes unequal length sequences
rnnoutput = lstm([[1,2,3,4],[5,6]];sorted=true,hy=true,cy=true)

#To see and modify rnn params in a structured view
lstm.gatesview
```


## TO-DO
3) Examples
4) Special layers such Google's `inception`   
5) Known embeddings such `Gloove`   
6) Pretrained Models   
