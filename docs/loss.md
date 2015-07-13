## Loss Layers

A LossLayer is typically the final layer of a feed forward neural
network.  It does not change the output of the network: its forw
function does nothing except record the network output y.  Its back
function takes z, the desired output, and overwrites it with the loss
gradient wrt y.  It also provides a loss function loss(z), which takes
the desired output, and returns a loss value.  Unless otherwise
specified their constructor takes no arguments, e.g. QuadLoss().

In the following discussion y and z are used for model and desired
outputs, q and p are used for model and desired probabilities when the
outputs are normalized.  When tensors are used normalization is across
the last dimension, i.e. sum(p[:,...,:,i])==1.  J is the loss
function, ∂J/∂yk is its derivative wrt the k'th output and ∇J(y) is
the loss gradient wrt y.  We sometimes use just dy as a short form for
∇J(y).

Cross entropy is a commonly used loss function for classification
problems.  Assume q is the class probabilities output by the model and
p is the desired probability vector (p is typically a one-hot vector
when we know the class for sure, but does not have to be).  Then the
cross-entropy loss is defined as:

    J(q) = -Σ pi log qi

In order to compute model probabilities q from raw network outputs y
one uses the softmax function:

    qi = (exp yi) / (Σ exp yj)

This is an expensive operation and it is often not needed during
prediction (argmax qi = argmax yi).  In those cases it would be better
to avoid normalization during forward computation and implement it
during the backward step of training.  To give users some flexibility,
KUnet implements three different cross entropy loss layers:

* [SoftLoss](https://github.com/denizyuret/KUnet.jl/blob/master/src/softloss.jl):
  assumes the network outputs are normalized probabilities, computing
  the loss and the gradients without performing softmax.  Should only
  be used after the Soft final layer, which applies softmax to the
  network output.  If q is the (normalized) network output and p is
  the desired probability vector:

        J(q) = -Σ pi log qi
             = -Σ pi log (qi/Σqj)    ;; should make normalization explicit
             = (-Σ pi log qi) + Σ pi log Σ qj
             = (-Σ pi log qi) + log Σ qj

        ∂J/∂qk = -pk/qk + (1/Σ qj)
               = -pk/qk + 1

* [XentLoss](https://github.com/denizyuret/KUnet.jl/blob/master/src/xentloss.jl):
  assumes the network outputs are raw (unnormalized log probabilities)
  and performs softmax before computing the loss and its gradients.
  Can be used after any final layer.  If y is the network output, q is
  softmax applied to y, and p is the desired probability vector:

        J(y) = -Σ pi log qi
             = -Σ pi log ((exp yi) / (Σ exp yj))
             = -(Σ pi yi) + (Σ pi log Σ exp yj)
             = -(Σ pi yi) + (log Σ exp yj)

        ∂J/∂yk = -pk + (exp yk) / (Σ exp yj)
               = -pk + qk

* [LogpLoss](https://github.com/denizyuret/KUnet.jl/blob/master/src/logploss.jl):
  assumes the network outputs are normalized log probabilities and
  computes the loss and the gradients accordingly.  Log probability
  outputs are sometimes useful when computing log likelihoods.  Should
  only be used after the Logp final layer, which applies log-softmax
  to the network output.  The loss and the gradient are identical to
  XentLoss except Σ exp yj = 1 is assumed.

Note that Net+XentLoss, Net+Soft+SoftLoss, Net+Logp+LogpLoss
essentially implement the same network for training purposes.  They
only differ in the efficiency and the normalization of their forward
output.

Other loss layers implemented are:

* [QuadLoss](https://github.com/denizyuret/KUnet.jl/blob/master/src/quadloss.jl):
  quadratic loss layer, can be used after any final layer.

        J(y) = (1/2) Σ (yi-zi)^2
        ∂J/∂yk = yk - zk.

* [PercLoss](https://github.com/denizyuret/KUnet.jl/blob/master/src/percloss.jl):
  implements the perceptron loss function.  A multiclass perceptron
  can be constructed using an Mmul layer followed by PercLoss and
  using a learning rate of 1.  If i is the correct answer and j is the
  predicted answer, the perceptron loss is:

        J(y) = -yi + yj
        ∂J/∂yk = -(k=i) + (k=j)

    Here is why Mmul+PercLoss is identical to a perceptron: For a
    given column with input x, the multiclass perceptron update rule
    is:

        w[i,:] += x;  w[j,:] -= x

    Note that there is no update if cz==cy.  The Mmul updates are:

        dw = dy*x'
        dx = w'*dy

    So the perceptron update will be performed by Mmul if we pass a dy
    matrix back where in each column we have all zeros if the
    predicted answer is correct, otherwise the correct answer is
    marked with -1 and the predicted answer is marked with a +1.
    Think of this as the gradient of the loss, i.e. going in this
    direction will increase the loss.

    This is an experimental layer for investigating how perceptron
    learning can be integrated in a multilayer network framework.  I
    recommend using the stand-alone Perceptron layer for real work.
