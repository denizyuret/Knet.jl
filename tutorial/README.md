# Knet Tutorial

This tutorial introduces the programming language Julia and the Knet deep learning
framework. By the end, the reader should be able to define, train, evaluate, and visualize
basic MLP, CNN, and RNN models.  Each notebook is written to work stand-alone but they rely
on concepts introduced in earlier notebooks, so I recommend reading them in order. Every
Knet function outside of the standard Julia library is defined or explained before use.

To run the notebooks on your computer, install and run IJulia by typing the following at the
`julia>` prompt (see [IJulia.jl](https://github.com/JuliaLang/IJulia.jl) for more
information):

```julia-repl
julia> using Pkg; Pkg.add("IJulia"); Pkg.add("Knet")
julia> using IJulia, Knet
julia> notebook(dir=Knet.dir("tutorial"))
```

To run the notebooks in the cloud you can use [JuliaBox](), [Google
Colab](https://colab.research.google.com/notebooks/welcome.ipynb), or services like
[AWS](http://aws.amazon.com). To run on JuliaBox, click the Git button in the Dashboard and
clone `https://github.com/denizyuret/Knet.jl.git`. The tutorial should be available under
`Knet/tutorial` on the Jupyter screen.  To run on Colab add Julia support first using the
[colab_install_julia](colab_install_julia.ipynb) notebook, then open the notebooks in
[Google
Drive](https://drive.google.com/drive/folders/19D-R31unxZV_PUYYYpCfd-gnbdUiZfNb?usp=sharing).
To run on AWS follow the instructions in the [Knet Installation
Section](http://denizyuret.github.io/Knet.jl/latest/install.html#Using-Amazon-AWS-1).

**Contents:**
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
