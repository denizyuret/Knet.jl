# Reference

**Contents**

```@meta
CurrentModule = Knet
```

```@contents
Pages = ["reference.md"]
```

## AutoGrad

```@docs
AutoGrad
```

## KnetArray

```@docs
Knet.KnetArray
```

## File I/O
```@docs
Knet.save
Knet.load
Knet.@save
Knet.@load
```

## Parameter initialization

```@docs
Knet.param
Knet.xavier
Knet.xavier_uniform
Knet.xavier_normal
Knet.gaussian
Knet.bilinear
```

## Activation functions
```@docs
Knet.elu
Knet.relu
Knet.selu
Knet.sigm
```

## Loss functions
```@docs
Knet.accuracy
Knet.bce
Knet.logistic
Knet.logp
Knet.logsoftmax
Knet.logsumexp
Knet.nll
Knet.softmax
Knet.zeroone
```

## Convolution and Pooling

```@docs
Knet.conv4
Knet.deconv4
Knet.pool
Knet.unpool
```

## Recurrent neural networks

```@docs
Knet.RNN
Knet.rnnparam
Knet.rnnparams
```

## Batch Normalization

```@docs
Knet.batchnorm
Knet.bnmoments
Knet.bnparams
```

## Model optimization

```@docs
Knet.minimize
Knet.converge
Knet.minibatch
Knet.progress
Knet.training
```

## Hyperparameter optimization

```@docs
Knet.goldensection
Knet.hyperband
```

## Utilities

```@docs
Knet.bmm
AutoGrad.cat1d
Knet.cpucopy
Knet.dir
Knet.dropout
Knet.gc
Knet.gpu
Knet.gpucopy
Knet.invx
Knet.mat
Knet.seed!
```

## AutoGrad (advanced)

```@docs
AutoGrad.@gcheck
AutoGrad.@primitive
AutoGrad.@zerograd
```

## Per-parameter optimization (advanced)

The model optimization methods apply the same algorithm with the same configuration to every
parameter. If you need finer grained control, you can set the optimization algorithm and
configuration of an individual `Param` by setting its `opt` field to one of the optimization
objects like `Adam` listed below. The `opt` field is used as an argument to `update!` and
controls the type of update performed on that parameter. Model optimization methods like `sgd`
will not override the `opt` field if it is already set, e.g. `sgd(model,data)` will perform an
`Adam` update for a parameter whose `opt` field is an `Adam` object. This also means you can
stop and start the training without losing optimization state, the first call will set the
`opt` fields and the subsequent calls will not override them.

```@docs
Knet.update!
Knet.SGD
Knet.Momentum
Knet.Nesterov
Knet.Adagrad
Knet.Rmsprop
Knet.Adadelta
Knet.Adam
```

## Function Index

```@index
Pages = ["reference.md"]
```
