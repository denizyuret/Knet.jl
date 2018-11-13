# Knet

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://denizyuret.github.io/Knet.jl/latest)
[![](https://travis-ci.org/denizyuret/Knet.jl.svg?branch=master)](https://travis-ci.org/denizyuret/Knet.jl)
[![](https://gitlab.com/JuliaGPU/Knet.jl/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/Knet.jl/pipelines)
[![](https://coveralls.io/repos/github/denizyuret/Knet.jl/badge.svg?branch=master)](https://coveralls.io/github/denizyuret/Knet.jl?branch=master)
[![](https://codecov.io/gh/denizyuret/Knet.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/denizyuret/Knet.jl)

[Knet](https://denizyuret.github.io/Knet.jl/latest) (pronounced "kay-net") is the [Koç
University](http://www.ku.edu.tr/en) deep learning framework implemented in
[Julia](http://docs.julialang.org) by [Deniz Yuret](http://www.denizyuret.com) and
collaborators.  It supports GPU operation and automatic differentiation using dynamic
computational graphs for models defined in plain Julia. You can install Knet with the
following at the julia prompt: `using Pkg; Pkg.add("Knet")`. Some starting points:

* [Tutorial:](tutorial)
  introduces Julia and Knet via examples. Currently the most up-to-date resource, some of the documentation and examples below may be out of date.
* [Documentation:](https://denizyuret.github.io/Knet.jl/latest)
  installation, introduction, design, implementation, full reference and deep learning chapters.
* [Examples:](examples)
  more tutorials and example models.
* [Benchmarks:](http://denizyuret.github.io/Knet.jl/latest/tutorial.html#Benchmarks-1)
  comparison of Knet's speed with TensorFlow, PyTorch, DyNet etc.
* [Paper:](https://goo.gl/zeUBFr)
  Yuret, D. "Knet: beginning deep learning with 100 lines of julia." In *Machine Learning Systems Workshop* at NIPS 2016.
* [KnetML:](https://github.com/KnetML)
  github organization with Knet repos of models, tutorials, layer collections and other resources.
* [Images:](http://denizyuret.github.io/Knet.jl/latest/install.html#Using-Amazon-AWS-1)
  Knet machine images are available for [AWS](http://denizyuret.github.io/Knet.jl/latest/install.html#Using-Amazon-AWS-1), [Singularity](https://github.com/KnetML/singularity-images) and [Docker](https://github.com/JuliaGPU/docker).
* [Issues:](https://github.com/denizyuret/Knet.jl/issues)
  if you find a bug, please open a github issue.
* [knet-users:](https://groups.google.com/forum/#!forum/knet-users)
  if you need help or would like to request a feature, please join this mailing list.
* [knet-dev:](https://groups.google.com/forum/#!forum/knet-dev)
  if you would like to contribute to Knet development, please join this mailing list and check out these [tips](http://denizyuret.github.io/Knet.jl/latest/install.html#Tips-for-developers-1).
* [slack:](https://julialang.slack.com)
  there also exists #knet channel on [JuliaLang slack workspace](https://julialang.slack.com) (click [here](https://slackinvite.julialang.org) to get an invite).
* Related work: Please check out [Flux](https://github.com/FLuxML), [Mocha](https://github.com/pluskid/Mocha.jl), [JuliaML](https://github.com/JuliaML), [JuliaDiff](https://github.com/JuliaDiff), [JuliaGPU](https://github.com/JuliaGPU), [JuliaOpt](https://github.com/JuliaOpt) for related packages.

## Example

Here is a simple example where we define, train and test the
[LeNet](http://yann.lecun.com/exdb/lenet) model for the
[MNIST](http://yann.lecun.com/exdb/mnist) handwritten digit recognition dataset from scratch
using 13 lines of code and 30 seconds of GPU computation.

```julia
using Knet

# Define convolutional layer:
struct Conv; w; b; f; end
(c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
Conv(w1,w2,cx,cy,f=relu) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f)

# Define dense layer:
struct Dense; w; b; f; end
(d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b)
Dense(i::Int,o::Int,f=relu) = Dense(param(o,i), param0(o), f)

# Define a chain of layers:
struct Chain; layers; end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

# Define the LeNet model
LeNet = Chain((Conv(5,5,1,20), Conv(5,5,20,50), Dense(800,500), Dense(500,10,identity)))

# Train and test LeNet (about 30 secs on a gpu to reach 99% accuracy)
include(Knet.dir("data","mnist.jl"))
dtrn, dtst = mnistdata()
train!(LeNet, dtrn)
accuracy(LeNet, dtst)
```

## Contributing

Knet is an open-source project and we are always open to new contributions: bug reports and
fixes, feature requests and contributions, new machine learning models and operators,
inspiring examples, benchmarking results are all welcome.

Current contributors:

  * Can Gümeli
  * Carlo Lucibello
  * Ekin Akyürek
  * Ekrem Emre Yurdakul
  * Emre Ünal
  * Emre Yolcu
  * Enis Berk
  * Erenay Dayanık
  * İlker Kesen
  * Meriç Melike Softa
  * Mike Innes
  * Onur Kuru
  * Ozan Arkan Can
  * Ömer Kırnap
  * Phuoc Nguyen
  * Rene Donner
  * Tim Besard
  * Zhang Shiwei
