# Knet.Ops21: the Ops21 operator set for deep learning

Knet operators circa 2021 are collected in the Ops21 submodule.  Operators are stateless
functions that extend the base language with machine learning specific functions such as
convolutions, losses, activation functions etc. Other operator sets (e.g. NNlib, Keras etc)
can live in their own submodules. Model implementations that use a specific operator module
are hopefully guaranteed not to break in the future as long as they import
e.g. Knet.Ops21. Instead of breaking Ops20, I make changes in Ops21.

This submodule has documentation, generic typeless implementations and gradient
definitions. Ops21 functions can/should have array-type specific implementations in
array-type specific submodules.  KnetArray and CuArray implementations are in Knet.Ops21_gpu
(which does not add any functions itself, only provides GPU implementations). Some
implementations may be imported from other packages such as NNlib.

The following principles guide the API design:

* Choose a minimal set of functions that enable efficient implementations of SOTA models.
* Prefer functional programming: weights and state are passed as arguments and returned as values.
* Prefer keyword arguments, leave large structs with weights and state to Layers.
* Stay close to the CUDNN API.

Exported function list:

* **activation:** elu, gelu, hardsigmoid, hardswish, relu, selu, sigm, swish
