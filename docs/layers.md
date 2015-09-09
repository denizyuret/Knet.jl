## Layers

A layer represents a primitive function (e.g. matrix multiplication,
activation function) in KUnet.  One design decision in KUnet has been
to define layers as fine grained as possible (e.g. separate mmul,
bias, relu into their own layers) to reduce the number of
configuration options an facilitate code reuse.  Here is a list of
layers implemented:

* [Mmul](https://github.com/denizyuret/KUnet.jl/blob/master/src/mmul.jl), [Bias](https://github.com/denizyuret/KUnet.jl/blob/master/src/bias.jl): matrix multiplication and bias for feed forward nets.
* [Conv](https://github.com/denizyuret/KUnet.jl/blob/master/src/conv.jl), [Pool](https://github.com/denizyuret/KUnet.jl/blob/master/src/pool.jl): convolution and pooling for convolutional nets.
* [Add2](https://github.com/denizyuret/KUnet.jl/blob/master/src/add2.jl), [Mul2](https://github.com/denizyuret/KUnet.jl/blob/master/src/mul2.jl): elementwise addition and multiplication for recurrent nets.
* [Activation Layers](actf.md) implement activation functions, e.g. sigmoid, tanh and relu.
* [Loss Layers](loss.md) implement loss functions, e.g. cross entropy and quadratic loss.
* [Drop](https://github.com/denizyuret/KUnet.jl/blob/master/src/drop.jl): dropout layer.
* [Perceptrons](perceptron.md) describes layers for perceptrons and kernel perceptrons.

Feed forward, convolutional, recurrent nets and perceptrons are
constructed by gluing together layers.  For the glue to work, each
layer has to follow a common interface.  This document describes this
common Layer interface.

### Storage



### Forward calculation

`forw(l::Layer, x)` takes input x and returns output y, possibly reading and/or writing some internal state.  
For layers with more than one input (e.g. Add2, Mul2), use `forw(l, x1, x2)`.  `ninputs(l::Layer)` will tell you how many inputs a layer takes.
Each layer supports one or more of the following array types for x: Array, CudaArray, SparseMatrixCSC, KUdense, KUsparse.
The input x can also be nothing, representing the zero matrix: currently supported by Add2, Mul2, Mmul.  If the input is nothing, Mul2 and Mmul return nothing, Add2 returns its other argument.
The return array y will be of the same type as x, except when x is sparse.  
If x is SparseMatrixCSC, y will be an Array, if x is KUsparse, y will be KUdense with the same location (cpu/gpu) and precision.  
The layer is responsible for allocating space if necessary for its output y.  
To conserve memory several strategies have been implemented:

* `overwrites(l::Layer)=>true`: Some layers overwrite and return their inputs:
  - Bias, Drop, and Activation layers overwrite the input array x.
  - Loss layers simply return x without touching it.
  - Add2 overwrites its second argument.
* `overwrites(l::Layer)=>false`: Other layers allocate new space for their output:
  - Mmul, Conv, Pool need new space because their output size may be different than their input.
  - Mul2 needs new space because the two inputs are needed for the gradient calculation.
* When new space is needed for the output, here is how we proceed:
  - If a y keyword argument is specified, it is considered first.
  - If a y array has been used in a previous call it is considered next.
  - If the y under consideration has the right type but wrong size, it is resized.
  - If all else fails a new y of the right type and size is allocated.

### Backward calculation

`back(l::Layer, dy)` Let x be the forward input, y the forward output, and w layer weights, if any.
Let dx, dy, dw stand for the loss gradient with respect to x, y, w.
Loss layers are an exception: they interpret dy as the desired output.
Back takes dy and returns dx, calculating dw as a side effect.
Only Mmul, Conv, and Bias have parameters and calculate dw (you can use param(l::Layer)!=nothing to test whether a layer has parameters).  
The keyword argument `returndx` can be set to false to prevent dx calculation for input layers for efficiency.

Layers that overwrite x with y going forward also overwrite dy with dx
going backward.  Non-overwriting layers allocate their own space for
dx, following a similar procedure used to allocate y.  The keyword
argument `dx` can be used to provide space for the output.

Some layers have dependencies other than dy for backward calculation:

* Conv, Mmul, Mul2 need the input x.
* Activation and loss layers need output y.
* Pool needs both input x and output y.
* Add2, Bias, and Drop do not need anything other than dy.

You can find out whether a layer reads x and/or y going back using
`back_reads_x(l::Layer)`, `back_reads_y(l::Layer)`.  By default a
layer will record the last x and/or y going forward and use it for the
backward calculation.  However for RNNs, where the same layer can be
entered multiple times, keyword arguments `x` and `y` can be used to
specify the particular forward input and output to use during backward
calculation.

<!---

# DEAD TEXT

TODO: rename layer -> op

TODO: mention keyword args forw/y and back/dx.

TODO: consider allocating layers rather than matrices for the RNN.
Have to figure out weight sharing for Mmul, Conv, and Bias.

TODO: write static rnn analyzer and give warnings.

TODO: rethink the layer/net dichotomy: allow for compound layers such
as lstm, possibly create a recursive data structure that can
encapsulate rnn and net (however rnn keeps track of history but net
does not, on the other hand net can be seen as a single time step
rnn).  Allowing easy construction of high level abstractions.

TODO: implement dropout using mul2.

TODO: rethink param(l) interface.

TODO: implement and document needx, needy.

TODO: specify which layer supports which array

TODO: 


Feed forward networks recall the last x for each layer by recording it in l.x.
They overwrite dx and dw as necessary.

RNNs cannot do this:
The same layer gets called on many different x's during the forward pass.
- We need to specify the x for back.
- dw should be initialized to zero and cumulatively incremented.
There may be multiple inputs to the layer:
- Add2 and Mul2 have two x's.
The output of the layer may be used by more than one other layer:
- dx should be initialized to zero and cumulatively incremented for such layers.
Layers overwrite their inputs:
- we have lost the original x's.
How to accept preallocated dx's:
- we have multiple forw calls, so a single dx array won't cut it.

Can we do all this preserving the feed forward interface and avoiding
unnecessary allocation and initialization?

Going forward we have the option of overwriting x with y.
Going back we have the option of overwriting dy or x with dx.

Overwriting good for two reasons:
- avoid allocation, use less space.
- avoid copying!?  add2/dx, bias/dx, loss/y

Can we automatically detect garbage blobs and reuse? When does a blob become garbage?
When nobody else is going to use it in the future:
- forw this time step
- forw future time steps
- back

Going forward all layers need is their x, so we have a problem only when x's are reused.
i.e. if an x is used once, it becomes garbage as soon as y is computed.

Going back layers may need:
- dy: add2, bias, drop
- dy,x: conv, mmul, mul2, loss
- dy,y: actf?
- dy,x,y: pool?

Going back dx and dw may have to be incremented multiple times.
Going forward all y is written only once.
We have a tree going forward: each blob is written by a unique layer[i,t]
then possibly overwritten with reuse.

Overwriting x going forward is bad because x may be later needed by:
- you going backward this time step?
- another layer going forward with the same input?
- previous layer needs its y going back.

Overwriting dy going back is bad when:
- multiple incremental updates are needed when multiple outputs: 
-- mmul and mul2 are siblings with input from actf, mmul, data.
- separate dx are needed for multiple inputs: mul2,add2 neighter modify dy.

Reusing x for dx is bad when:

Overwriting dw going back is bad when:
- always: each time step will incrementally update dw.
=> modify mmul, conv, bias so dw can be reset or incremented.


For each cell compute: inputs, forwdeps, backdeps.
inputs[n] are the cells forw net[n] uses to calculate cell[n].
forwdeps[n] are the cells that use cell[n] in forw calculation.
backdeps[n] are the cells that use cell[n] in back calculation.

Forw overwrite x is ok if nobody else is going to use x.
forwdeps[x] should be a singleton
backdeps[x] should be empty

Back overwrite dy is ok unless:
- mul2 needs to send it back to two inputs.
- add2 duplicates it so there is shared memory going back.
- if forwdeps > 1 we will have multiple incremental updates.

Back overwrite x is ok unless:
- that x is somebody's y, which they may need.
- check to see if backdeps[x] is empty.
- why did we never use this?

This is like immediate gc().

|x|==|y| for actf,bias,drop,loss,add2,mul2
|x|!=|y| for conv,mmul,pool

Overwrite x  going forw is only an option if |x|==|y|.  All except mul2 overwrite.
Overwrite dy going back is only an option if |x|==|y|.
Overwrite x going back is always an option. (unless parent needs y,dy).
only actf and pool need y.

How about using the same y grid to compute dy?

Ask the user to insert copy layers?

Can back for mmul overwrite x?

--->
