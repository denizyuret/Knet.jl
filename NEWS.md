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

Documentation and Testing
-------------------------
* New links added to opt and rl docs.

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
