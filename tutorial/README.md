# Knet Tutorial

This tutorial introduces the programming language Julia and the Knet deep learning
framework. By the end, the reader should be able to define, train, evaluate, and visualize
basic MLP, CNN, and RNN models.  Each notebook is written to work stand-alone but they rely
on concepts introduced in earlier notebooks, so I recommend reading them in order. Every
Knet function outside of the standard Julia library is defined or explained before use.

To run the notebooks on a Jupyter server, start julia in this directory then install and run
IJulia by typing the following at the `julia>` prompt: (see
[IJulia.jl](https://github.com/JuliaLang/IJulia.jl) for more information).

    julia> using Pkg
    julia> Pkg.add("IJulia")
    julia> using IJulia
    julia> notebook()

You can also just view the notebooks on github and type the examples manually at a Julia
prompt. The later examples require a GPU machine to train in a reasonable amount of time.

**Contents:**
* [00.Julia_is_fast:](00.Julia_is_fast.ipynb) comparison of Julia's speed to C, Python and numpy.
* [01.Getting_to_know_Julia:](01.Getting_to_know_Julia.ipynb) basic Julia tutorial from [juliabox.com](http://juliabox.com).
* [02.mnist:](02.mnist.ipynb) introduction to the MNIST handwritten digit recognition dataset.
* [03.lin:](03.lin.ipynb) define, train, visualize simple linear models, introduce gradients, SGD, using the GPU.
* [04.mlp:](04.mlp.ipynb) multi layer perceptrons, nonlinearities, model capacity, overfitting, regularization, dropout.
* [05.cnn:](05.cnn.ipynb) convolutional neural networks, sparse and shared weights using conv4 and pool operations.
* [06.rnn:](06.rnn.ipynb) introduction to recurrent neural networks.
* [07.imdb:](07.imdb.ipynb) a simple RNN sequence classification model for sentiment analysis of IMDB movie reviews.
* [08.charlm:](08.charlm.ipynb) a character based RNN language model that can write Shakespeare sonnets and Julia programs.
* [09.s2s:](09.s2s.ipynb) a sequence to sequence RNN model typically used for machine translation.
