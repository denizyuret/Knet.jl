Knet v0.8.2 Release Notes
=========================

General
-------
* Fixed bug in (scalar .> KnetArray) operation in cuda01.jl.
* Added optional w arg to Sgd for consistency in update.jl.
* KnetArray supports transpose and permutedims for 2-D and 3-D (@ekyurdakul).
* cpu2gpu, gpu2cpu primitives with gradients in karray.jl (@ereday).
* gaussian, xavier, bilinear initializers in distributions.jl (@ekyurdakul).
* New deconvolution and unpool operations in cuda44.jl (@ekyurdakul).
* KnetArray now supports isempty, (.!=), and some more math functions like sin,log etc.
* Default conv4 padding changed back to 0. Knet.padsize(w) still available to compute input size preserving padding.
* GPU garbage collection does not print '.', '+' unless user modifies gcinfo macro.
* It is now an error to try to create a KnetArray when the active device is not a gpu.
* CPU convolution and pooling operations implemented (@kuruonur1).
* Updated AutoGrad requirement to v0.0.5.

Documentation and Testing
-------------------------
* New links added to opt and rl docs.
* gradcheck automatically handles non-scalar functions, stays quiet by default, returns true/false. Moved to AutoGrad.
* All documentation moved to Markdown using Documenter.jl.
* Documentation for examples and a reference section added.
* Unsupported ops documented in KnetArray doc.

Examples
--------
* optimizers.jl padding bug fixed.
* vgg.jl now supports D and E models (@ilkerkesen).
* Batch Normalization/ResNet example (resnet.jl) added. (@ilkerkesen)


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
