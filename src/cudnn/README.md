## Knet.CUDNN: High level interface to cuDNN functions

The goal of this submodule is to map the low level cuDNN calls to more natural Julia
functions. Here are some design choices:

**Naming:** We try to keep the same function, argument, and type names from the cuDNN
library in the high level interface. The wrappers for descriptors drop the `_t` suffix,
e.g. `cudnnPoolingDescriptor_t => cudnnPoolingDescriptor`.

**Descriptors:** The cuDNN functions take data and operator descriptors. Most of these
descriptors are relatively fast to create (~500 ns for a cudnnTensorDescriptor) so they may
not be worth preallocating. We also use caching (~100 ns) to save a bit of memory and
speed. All descriptor fields are `isbits` types with the exception of the
`cudnnDropoutDescriptor` which points to a random number generator state and is used as a
field of some other descriptors.

**Operator descriptors:** Descriptors such as `cudnnPoolingDescriptor` specify the options
for an operator such as stride and padding. For operators with descriptors we have one
method that takes keyword arguments with reasonable defaults to construct the descriptor and
another method that takes a pre-initialized descriptor as its last argument.  This way a
casual user can call the first method without worrying about the descriptor format, only
specifying non-default options, whereas a layer architect can keep a preset descriptor in
the layer that gets passed to the function using the second method.

**Output arrays:** The cuDNN functions take pre-allocated output arrays. We will have a
Julia function that allocates its own output array (e.g. `cudnnPoolingForward`) and a bang
version that takes a pre-allocated output array as the first argument
(e.g. `cudnnPoolingForward!`).

**Methods:** Each cuDNN forward function may have up to four methods depending on whether
the descriptor and the output array is specified:

    cudnnPoolingForward(x)
    cudnnPoolingForward(x, poolingDesc)
    cudnnPoolingForward!(y, x)
    cudnnPoolingForward!(y, x, poolingDesc)
