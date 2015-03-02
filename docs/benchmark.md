## Benchmark

KUnet contains a complete backprop implementation in
[Matlab](https://github.com/denizyuret/KUnet.jl/tree/master/matlab)
and pure
[C/CUDA](https://github.com/denizyuret/KUnet.jl/tree/master/cuda) as
well as
[Julia](https://github.com/denizyuret/KUnet.jl/tree/master/src).  I
use these (and [Caffe](http://caffe.berkeleyvision.org)) for
debugging and benchmarking.

Here are the timing results for my standard backprop test with:
* dataset of 76834 instances in 1326 dimensions and 3 classes.
* a network with a single hidden layer of 20000 units.
* 1 epoch with 82 minibatches of 937 instances each.
* standard SGD (no momentum) with learningRate=0.01.
* a server with Tesla K20m GPU.

Implementation | Seconds/Epoch
---------------|--------------
Matlab| 7.95
Caffe | 6.76
Julia | 5.52
Cuda  | 4.87
