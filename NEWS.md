Knet v0.8.3 Release Notes
=========================

General
-------
* KnetArray indexing support for Int, Colon, UnitRange, StepRange, CartesianIndex, Array{Int}, Array{Bool}, Array{CartesianIndex}. Most combinations work for 2-D.  N-D indexing incomplete.  See `@doc KnetArray` for details.
* KnetArray multi-argument `hcat` and `vcat` implemented.
* Implemented `hyperband` and `goldensection` hyperparameter optimization algorithms.
* Added per weight gradient clip to `update!`.
* Fixed `update!` issues with `grad::Void` and other mismatched `w,grad` types.
* Added `setseed` to do `srand` in both cpu and gpu.

Documentation and Testing
-------------------------
* RNN chapter and IJulia notebook added.
* Solutions to installation problems documented.

Examples
--------
* Fixed vgg and resnet demos to use the new version of Images.jl.
* Added prof/s2s.jl, a sequence-to-sequence RNN model, for profiling.
* Added prof/karray.jl, profiling concatenation.


Knet v0.8.2 Release Notes
=========================

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
