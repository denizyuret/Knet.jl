# KUnet

* [Installation](docs/install.md)
* [Usage](docs/usage.md)
* [Code tutorial](http://www.denizyuret.com/2015/02/beginning-deep-learning-with-500-lines.html)

KUnet.jl is the beginnings of a deep learning package for Julia with emphasis on conciseness, clarity and easy extendability. It started as a challenge to see how many lines of (readable) code were sufficient to express deep learning algorithms given the right language.  A secondary concern was efficiency: being able to run the same code on GPU with minimal trouble.  Currently, only the basic functionality is in place (i.e. backprop with relu, softmax, sgd, momentum, nesterov, adagrad, dropout, l1-l2 regularization etc.) but the GPU functionality is in, its speed is competitive with [Caffe](http://caffe.berkeleyvision.org/), and I think convolutional and recurrent nets can be added without too much effort.  I wrote a [blog post](http://www.denizyuret.com/2015/02/beginning-deep-learning-with-500-lines.html) about the code structure and there is some basic documentation here.  You can send me suggestions for improvement (both in coding style and new functionality) using [comments](http://www.blogger.com/comment.g?blogID=8540876&postID=328231440874481473) to the [blog post](http://www.denizyuret.com/2015/02/beginning-deep-learning-with-500-lines.html), or using [issues](https://github.com/denizyuret/KUnet.jl/issues) or [pull requests](https://help.github.com/articles/fork-a-repo/) on GitHub.

I tried to make the code (cpu/gpu) generic and close to how we think of these algorithms mathematically.  Getting the same code working on the GPU and the CPU in Julia proved to be a bit challenging and showed that both a more standard treatment of CPU and GPU arrays, and a standard syntax for in-place operations would be welcome additions to the language.  I'd like to thank Tim Holy ([CUDArt](https://github.com/JuliaGPU/CUDArt.jl)), Nick Henderson ([CUBLAS](https://github.com/JuliaGPU/CUBLAS.jl)), and Simon Byrne ([InplaceOps](https://github.com/simonbyrne/InplaceOps.jl)) for their generous help.

### Related Links
* [Mocha.jl](https://github.com/pluskid/Mocha.jl): a deep learning framework for Julia
* [BackpropNeuralNet.jl](https://github.com/compressed/BackpropNeuralNet.jl): another neural net implementation
* [UFLDL](http://ufldl.stanford.edu/tutorial/): deep learning tutorial
* [deeplearning.net](http://deeplearning.net/): resources and pointers to information about Deep Learning
* [Some starting points for deep learning](http://www.denizyuret.com/2014/11/some-starting-points-for-deep-learning.html)
