A deep learning library can be organized at three levels of abstraction: operators, layers,
models. Operators are stateless functions that extend the base language with machine
learning specific functions such as convolutions, losses, activation functions etc. Layers
are functions with weight arrays and/or internal state that form the building blocks of
models, some examples are RNNs, Transformers, Convolutional and Residual layers. Knet
implements layers as callable objects. Models combine layers to accomplish a specific task,
typically have loss, predict, train functionality, some examples are LeNet, Yolo, BERT,
ResNet etc.

Each of these levels have various implementations and evolve over time. Knet is redesigned
starting v1.4 to support multiple versions of multiple implementations simultaneously. A
related goal is to freeze sets of operators / layers periodically giving long term support
to models built with older operator / layer sets.

* **autograd_gpu:** implementations of AutoGrad functions for GPU arrays.
* **cuarrays:** implementations of Base functions for CuArrays.
* **knetarrays:** KnetArrays and their Base functions.
* **libknet8:** hand-written CUDA kernels.
* **ops20:** the Knet.Ops20 operator set.
* **ops20_gpu:** implementations of Ops20 operators for GPU arrays.
* **train20:** Knet.Train20 model training utilities.
