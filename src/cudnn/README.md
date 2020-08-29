## Knet.CUDNN: High level interface to cuDNN functions

The goal of this submodule is to map the low level cuDNN calls to more natural Julia
functions. Here are some design choices:

**Naming:** We try to keep the same function, argument, and type names from the cuDNN
library in the high level interface. The wrappers for descriptors drop the `_t` suffix,
e.g. `cudnnDropoutDescriptor_t => cudnnDropoutDescriptor`. In-place operations have an `!`
suffix added: `cudnnSetTensor!`, `cudnnScaleTensor!`, `cudnnAddTensor!`.

**Output arrays:** The cuDNN functions take pre-allocated output arrays. We will have an
optional keyword argument `y` for the output array so the user can provide one if they want
to, but the function will allocate a new one if none provided.

**Array descriptors:** The cuDNN functions take tensor and filter descriptors along with
pointers to their data. These descriptors are relatively fast to create (~500 ns) so they
are not worth exposing in the high level interface. We use caching (~100 ns) for more
efficiency.

**Operator descriptors:** The cuDNN functions take preinitialized descriptors such as
cudnnActivationDescriptor_t that specify the options for the operation such as convolution
stride or activation function. We will have keyword arguments in the forward function both
for the pre-allocated descriptor and for the individual options. This way a casual user can
call the function with default options without worrying about the descriptor format, whereas
a layer architect can keep a preallocated descriptor in the layer that gets passed to the
function. When both the options and the descriptor are specified the descriptor overrides
the other options.
