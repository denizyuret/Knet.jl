# Reference

**Contents**

```@contents
Pages = ["reference.md"]
```

## AutoGrad

```@docs
AutoGrad.grad
AutoGrad.gradloss
AutoGrad.gradcheck
```

## KnetArray

```@docs
Knet.KnetArray
```

## Utilities

```@docs
Knet.accuracy
Knet.dir
Knet.dropout
Knet.gpu
Knet.invx
Knet.gc
Knet.logp
Knet.logsoftmax
Knet.softmax
Knet.logsumexp
Knet.minibatch
Knet.nll
Knet.logistic
Knet.bce
Knet.relu
Knet.seed!
Knet.sigm
```

## Convolution and Pooling

```@docs
Knet.conv4
Knet.deconv4
Knet.mat
Knet.pool
Knet.unpool
```

## Recurrent neural networks

```@docs
Knet.rnninit
Knet.rnnforw
Knet.rnnparam
Knet.rnnparams
```

## Batch Normalization

```@docs
Knet.bnmoments
Knet.bnparams
Knet.batchnorm
```

## Optimization methods

```@docs
Knet.update!
Knet.optimizers
Knet.Adadelta
Knet.Adagrad
Knet.Adam
Knet.Momentum
Knet.Nesterov
Knet.Rmsprop
Knet.Sgd
```

## Hyperparameter optimization

```@docs
Knet.goldensection
Knet.hyperband
```

## Initialization

```@docs
Knet.bilinear
Knet.gaussian
Knet.xavier
```

## AutoGrad (advanced)

```@docs
AutoGrad.getval
AutoGrad.@primitive
AutoGrad.@zerograd
```

## Function Index

```@index
Pages = ["reference.md"]
```
