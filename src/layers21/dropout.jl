export Dropout
using Knet.Ops21: dropout

"""
    Dropout(rate)

References:
* [Srivastava et al. 2014](http://www.jmlr.org/papers/v15/srivastava14a.html) Dropout: A Simple Way to Prevent Neural Networks from Overfitting
* torch.nn.Dropout
* tf.keras.layers.Dropout
"""
struct Dropout; p; end

function (l::Dropout)(x)        # TODO: how about seed and force args?
    dropout(x, l.p) # TODO: dropout normalization does not depend on masks?
end
