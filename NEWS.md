Knet v0.8.6 Release Notes
=========================

TODO
----
* CUDNN: add batchnorm, test dropout, softmax etc. for speed.
* Modular interface.
* CUDAapi, windows compat, Tim Besard's CI.

General
-------
* Pre-0.6 Julia versions no longer supported.
* `rnninit` and `rnnforw` implement cudnn RNNs (with @cangumeli).
* `conv4` performance significantly improved using cudnnFind.
* `DBGFLAGS` and `PROFILING` constants defined in Knet.jl.
* `optimizers` creates optimization structs for the whole model.
* `dropout` now detects training mode automatically.
* `nll` returns negative log likelihood given score matrix and answer index vector.
* `accuracy` returns ratio of correct answers given score matrix and answer index vector.
* `minibatch(x,y,b)` returns a batch iterator.
* `knetgc` is now exported to cudaFree garbage collected pointers.
* Using CUDAapi and CUDAdrv in build.jl if installed.
* Got rid of the Combinatorics dependency in test.
* libnvidia-ml only used when available (it is not available in OSX).

Documentation and Examples
--------------------------
* New benchmarking results in tutorial.md and README.md (from @ilkarman).
* New under Knet/data: mnist.jl, cifar.jl, imdb.jl, gutenberg.jl, mikolovptb.jl.
* All examples updated to use the new RNNs and replaced/supported with IJulia notebooks.

Knet v0.8.5 Release Notes
=========================
8ea130e 2017-10-21

General
-------
* Fixed memory leak with certain broadcast kernels (@ilkerkesen).
* Fixed dropout efficiency bug introduced in 0.8.4.
* Added conditional support for SpecialFunctions.
* Added Nesterov optimizer (@CarloLucibello).
* Removed Compat dependency.
* Proper handling of concat KnetArrays of incompatible eltypes (#175).
* Fixed dotted function handling in Julia5 (#173).

Documentation and Examples
--------------------------
* Fixed julia6 compat problem in examples/mnist.jl.
* charlm.jl can now save generated text (@DoguD).
* Added fashion-mnist.jl example (@quaertym).
* Added missing `MNIST.loaddata()` to tutorial.jl.
* Fixed julia4 compat problem in examples/vgg.jl.


Knet v0.8.4 Release Notes
=========================
2a2cba3 on 2017-09-09

* Julia 0.6 compatibility fixes.
* Fixed compiler flags in Makefile for compatibility.
* charlm unicode character problem fixed.


Knet v0.8.3 Release Notes
=========================
48d4924 on 2017-05-18

General
-------
* KnetArray support for general broadcasting operations (@EnisBerk).
* KnetArray support for general reduction operations (@ilkerkesen).
* KnetArray support for permutedims up to 5D (@ekyurdakul).
* KnetArray indexing support for Int, Colon, UnitRange, StepRange, CartesianIndex, Array{Int}, Array{Bool}, Array{CartesianIndex}. Most combinations work for 2-D.  N-D indexing incomplete.  See `@doc KnetArray` for details.
* KnetArray support for multi-argument `hcat` and `vcat`.
* KnetArray support for saving to and loading from JLD files.
* Implemented `hyperband` and `goldensection` hyperparameter optimization algorithms.
* Added per weight-array gradient clip to `update!`.
* Fixed `update!` issues with `grad::Void` and other mismatched `w,grad` types.
* Fixed `update!` issues with `grad::Dict` missing keys.
* Added `setseed` to do `srand` in both cpu and gpu.
* Added `dropout(a,p)` as a Knet primitive.
* Implemented `mean(::KnetArray,r)`.

Testing and Benchmarking
------------------------
* Benchmarks added under prof/ for reduction, broadcast, concat, conv operations; rnnlm, s2s models.

Documentation and Examples
--------------------------
* RNN chapter and IJulia notebook added.
* Updated the CNN chapter.
* Solutions to installation problems documented.
* Fixed vgg and resnet demos to use the new version of Images.jl and to work on CPU-only machines. Fixed batch normalization bug in resnet. (@ilkerkesen)
* Fixed charlm demo to use indexing operations, Adam, and dropout.
* Added rnnlm demo.


Knet v0.8.2 Release Notes
=========================
8f77f85 on Feb 23, 2017

General
-------
* update! now supports iterators and dictionaries of weight arrays.
* CPU convolution and pooling operations implemented based on CPP kernels from Mocha.jl.
* gradcheck automatically handles non-scalar functions, stays quiet by default, returns true/false. Moved to AutoGrad.
* KnetArray now supports isempty, (.!=), (==), isapprox, copy!, scale!, deepcopy, more math functions (exp,log etc.), and KnetArray{T}(dims) constructors.
* KnetArray supports transpose and permutedims for 2-D and 3-D (@ekyurdakul).
* Gradients for KnetArray to Array conversion (@ereday).
* gaussian, xavier, bilinear initializers in distributions.jl (@ekyurdakul).
* New deconvolution and unpool operations in cuda44.jl (@ekyurdakul).
* Default conv4 padding changed back to 0. Knet.padsize(w) still available to compute size-preserving padding.
* GPU garbage collection does not print '.', '+' unless user modifies gcinfo macro.
* It is now an error to try to create a KnetArray when the active device is not a gpu.
* Fixed bug in (scalar .> KnetArray) operation in cuda01.jl.
* Updated AutoGrad requirement to v0.0.6.

Documentation and Testing
-------------------------
* Comprehensive unit-testing implemented.
* All documentation moved to Markdown using Documenter.jl.
* Documentation for examples and a reference section added.
* Unsupported ops documented in KnetArray doc.

Examples
--------
* Batch Normalization/ResNet example (resnet.jl) added. (@ilkerkesen)
* vgg.jl now supports D and E models (@ilkerkesen).
* hyperband.jl hyperparameter optimization example added.
* softmax.ipynb IJulia notebook about underfitting, overfitting, regularization and dropout.
* optimizers.jl padding bug fixed.


Knet v0.8.1 Release Notes
=========================
c9556d4 on Dec 7, 2016

General
-------

* update! defined supporting SGD, Momentum, Adagrad, Adadelta, Rmsprop, Adam (@ozanarkancan).
* Knet.dir(path) returns path relative to Knet root.
* axpy!, rand!, vec defined for KnetArray.
* relu, sigm, invx defined for scalars.
* Numerically stable sigm and logsumexp implemented.
* GPU discovery and selection made more robust using nvml.
* GPU memory management improved.
* Default conv4 padding changed to (w-1)/2 which preserves input size.
* Older versions of libcudnn now supported.

Documentation and Testing
-------------------------

* profmlp deprecates profile_kn for speed tests.
* Documentation updated with Knet8 examples.
* README updated with RNN example, benchmarks, architecture.

Examples
--------

* VGG image recognition demo added.
* Examples Pkg.add required packages automatically.
* housing example supports train/test ratio option.
* examples/optimizers.jl has usage examples of new update! functions (@ozanarkancan).
