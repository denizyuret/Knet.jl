## Loss Layers

A LossLayer is typically the final layer of a feed forward neural
network.  It does not change the output of the network: its forw
function does nothing except record the network output y.  Its back
function takes z, the desired output, and overwrites it with the loss
gradient wrt y.  Its loss function takes z, the desired output, and
returns a loss value.

In the following discussion y and z are used for model and desired
outputs, q and p are used for model and desired probabilities when the
outputs are normalized.  J is the loss function, ∂J/∂yk is its
derivative wrt the k'th output.

Cross entropy is a commonly used loss function for classification
problems.  Assume q is the class probabilities output by the model and
p is the actual probabilities (p is typically a one-hot vector when we
know the class for sure, but does not have to be).  Then the
cross-entropy loss is defined as:

    J(q) = -Σ pi log qi

In order to compute model probabilities q from raw network outputs y
one uses the softmax function:

    qi = (exp yi) / (Σ exp yj)

This is an expensive function and it is often not needed during
prediction (argmax qi = argmax yi).  In those cases it would be better
to avoid normalization during forward computation and implement it
during the backward step of training.  To give users some flexibility,
KUnet implements three different cross entropy loss layers:

* XentLoss: assumes the network outputs are raw (unnormalized log
  probabilities) and performs softmax before computing the loss and
  its gradients.  Can be used after any final layer.

* SoftLoss: assumes the network outputs are normalized probabilities,
  computing the loss and the gradients without performing softmax.
  Should only be used after the Soft final layer.

* LogpLoss: assumes the network outputs are normalized log
  probabilities and computes the loss and the gradients accordingly.
  Log probability outputs are sometimes useful when computing log
  likelihoods.  Should only be used after the Logp final layer.

Note that Net+XentLoss, Net+Soft+SoftLoss, Net+Logp+LogpLoss
essentially implement the same network for training purposes.  They
only differ in the efficiency and the normalization of their forward
output.

Other loss layers implemented are:

* QuadLoss: quadratic loss layer, can be used after any final layer.
    J(y) = (1/2) Σ (yi-zi)^2
    ∂J/∂yk = yk - zk.

* PercLoss: implements the perceptron loss function.  A multiclass
  perceptron can be constructed using an Mmul layer followed by
  PercLoss.  This is an experimental layer for investigating how
  perceptron learning can be integrated in a multilayer network
  framework.  I recommend using the stand-alone Perceptron layer for
  real work.