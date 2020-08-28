# Knet.Train20: model training utilities

This module exports a number of training, minibatching, initialization, hyperparameter optimization utilities.

Exported function list:

* **data processing:** minibatch
* **hyper-optimization:** goldensection, hyperband
* **progress visualization:** progress, progress!
* **parameter initialization:** gaussian, xavier, xavier_uniform, xavier_normal, bilinear, param, param0
* **parameter optimization:** minimize, minimize!, converge, converge!, update!, clone, optimizers
* **optimization algorithms:** SGD, sgd, sgd!, Momentum, momentum, momentum!, Nesterov, nesterov, nesterov!, Adagrad, adagrad, adagrad!, RMSprop, rmsprop, rmsprop!, Adadelta, adadelta, adadelta!, Adam, adam, adam!
