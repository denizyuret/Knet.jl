Deep learning libraries can be seen at three levels of abstraction: operators, layers,
models. Operators are stateless functions that extend the base language with machine
learning specific functions such as convolutions, losses, activation functions etc. Layers
are functions with weight arrays and/or internal state that form the building blocks of
models, some examples are RNNs, Transformers and Residual layers. Knet implements layers as
callable objects. Models combine layers to accomplish a specific task, typically have loss,
predict, train functionality, some examples are LeNet, Yolo, BERT, ResNet etc.

Each of these levels have various implementations and evolve over time. Knet is redesigned
to support multiple versions of multiple implementations simultaneously. The goal is to
freeze sets of operators / layers periodically giving long term support to models built with
older operator / layer sets.

* cuarrays: support code for CuArrays.
* cuda: hand-written CUDA kernels.
* data: commonly used datasets and data utilities.
* knetarrays: support code for KnetArrays.
* layers: building blocks for models.
* models: commonly used deep learning models.
* ops: building blocks for layers.
* train: training utilities.
