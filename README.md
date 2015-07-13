# KUnet

* [Installation](docs/install.md)
* [Usage](docs/usage.md)
* [Benchmark](docs/benchmark.md)
* [Layers](docs/layers.md)
* [Loss Layers](docs/loss.md)
* [Perceptrons and Kernel Perceptrons](docs/perceptron.md)

KUnet.jl is the beginnings of a deep learning package for Julia with emphasis on conciseness, clarity and easy extensibility. It started as a challenge to see how many lines of (readable) code were sufficient to express deep learning algorithms given the right language.  A secondary concern was efficiency: being able to run the same code on GPU with minimal trouble.  The latest version is less than 1000 lines of code and supports backprop in feedforward nets with convolution, pooling, and inner product layers with/without bias, relu, tanh, sigmoid activations, softmax and quadratic loss, optimization with sgd, momentum, nesterov, adagrad, dropout, L1-L2 regularization, on both CPU/GPU, with Float32/Float64 arrays of 1-5 dimensions.  Its speed is competitive with [Caffe](http://caffe.berkeleyvision.org/) ([here is a benchmark](docs/benchmark.md)), and I think recurrent and boltzmann nets can be added without too much effort.  

You can send me suggestions for improvement (both in coding style and new functionality) using [issues](https://github.com/denizyuret/KUnet.jl/issues) or [pull requests](https://help.github.com/articles/fork-a-repo/) on GitHub.

I tried to make the code (cpu/gpu) generic and close to how we think of these algorithms mathematically.  Getting the same code working on the GPU and the CPU in Julia proved to be a bit challenging and showed that both a more standard treatment of CPU and GPU arrays, and a standard syntax for in-place operations would be welcome additions to the language.  I'd like to thank Tim Holy ([CUDArt](https://github.com/JuliaGPU/CUDArt.jl)), Nick Henderson ([CUBLAS](https://github.com/JuliaGPU/CUBLAS.jl)), and Simon Byrne ([InplaceOps](https://github.com/simonbyrne/InplaceOps.jl)) for their generous help.

### Related Links
* [Beginning deep learning with 500 lines of Julia](http://www.denizyuret.com/2015/02/beginning-deep-learning-with-500-lines.html): my neural net tutorial based on v0.0.1 of KUnet.
* [Mocha.jl](https://github.com/pluskid/Mocha.jl): a deep learning framework for Julia
* [BackpropNeuralNet.jl](https://github.com/compressed/BackpropNeuralNet.jl): another neural net implementation
* [UFLDL](http://ufldl.stanford.edu/tutorial): deep learning tutorial
* [deeplearning.net](http://deeplearning.net): resources and pointers to information about Deep Learning
* [Some starting points for deep learning](http://www.denizyuret.com/2014/11/some-starting-points-for-deep-learning.html), and [some more](http://www.denizyuret.com/2014/05/how-to-learn-about-deep-learning.html): my blog posts with links
