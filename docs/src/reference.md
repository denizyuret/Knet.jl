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
Knet.cpu2gpu
Knet.gpu2cpu
```

## Utilities

```@docs
Knet.dir
Knet.gpu
Knet.logp
Knet.logsumexp
Knet.invx
Knet.relu
Knet.sigm
```

## Convolution

```@docs
Knet.conv4
Knet.pool
Knet.mat
Knet.deconv4
Knet.unpool
```

## Optimization

TODO: need blurb here about how optimization works, need to apply update! to individual weight arrays etc.

```@docs
Knet.update!
Knet.Sgd
Knet.Momentum
Knet.Adagrad
Knet.Adadelta
Knet.Rmsprop
Knet.Adam
```

## Initialization

```@docs
Knet.gaussian
Knet.xavier
Knet.bilinear
```

## AutoGrad (advanced)

TODO: blurb here about how AutoGrad works, what the `Rec` type is etc.

```@docs
AutoGrad.@primitive
AutoGrad.@zerograd
AutoGrad.getval
```

## Function Index

```@index
Pages = ["reference.md"]
```

