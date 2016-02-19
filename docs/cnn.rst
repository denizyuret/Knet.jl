*****************************
Convolutional Neural Networks
*****************************

.. Notes from: http://cs231n.github.io/convolutional-networks

.. discuss efficiency, reducing parameters reduces learning complexity
.. even though mlp is universal, learning weights for a cat-recognizer
.. would end up repeating weights.

.. karpathy says fully connected would have too many params and that
.. would lead to overfitting.  I think the problem is not overfitting,
.. an architecture that does the same job with fewer parameters can
.. learn from fewer examples and generalize better.  It would be a
.. better prior.  Maybe that is overfitting.  Can we do a simple
.. example with polynomials?  Theory from Bayes or SLT?

.. If detecting a horizontal edge is important at some location in the
.. image, it should intuitively be useful at some other location as
.. well due to the translationally-invariant structure of
.. images. There is therefore no need to relearn to detect a
.. horizontal edge at every one of the 55*55 distinct locations in the
.. Conv layer output volume.

.. TODO: theory lectures on Bayes (MacKay 21), SLT (PAC, VC dims), Regret
.. (Shalev-Schwartz), mistake bounds (perceptron).

.. the neurons in a layer will only be connected to a small region of
.. the layer before it, instead of all of the neurons in a
.. fully-connected manner.

.. Every filter is small spatially (along width and height), but
.. extends through the full depth of the input volume.

.. karpathy calls the 3rd dimension "depth".  This is a mistake, it
.. will get confusing when we get to 3D convolution.  It is better to
.. call this dimension "channels" or "filters" or "slices".

.. two explanations: the "each neuron connected to a small region" vs
.. convolution as a filter that scans the image.  The first ignores
.. the fact that the neurons also share weights.  Filter is better.
.. Still maybe give both pictures, the matrix/filter picture and the
.. neural network picture.  Here is a quote for correspondence:

.. Every entry in the output volume can thus also be interpreted as an
.. output of a neuron that looks at only a small region in the input
.. and shares parameters with neurons in the same activation map
.. (since these numbers all result from applying the same filter).

.. talking about 3D instead of 4D ignoring the minibatching at first
.. is better?  But the conv weights still have to be 4D.  Still, the
.. volume of activations is a nice picture.

.. talk about (1) dimensions thru ops, and (2) hyperparameters of
.. ops. (3) connectivities.

.. Do we describe backprop? for conv, pool, normalization?
.. The backward pass for a convolution operation (for both the data
.. and the weights) is also a convolution (but with spatially-flipped
.. filters). This is easy to derive in the 1-dimensional case with a
.. toy example (not expanded on for now).


.. hyperparameters control the size of the output volume: the depth,
.. stride and zero-padding.

.. We will refer to a set of neurons that are all looking at the same
.. region of the input as a depth column.

.. It is also nice to give 1D convolution examples.


.. normalization?  karpathy says they have fallen out of favor?  For
.. various types of normalizations, see the discussion in Alex
.. Krizhevsky's cuda-convnet library API.


.. add knet examples to this and other sections.

.. .. during backpropagation, every neuron in the volume will compute the
.. .. gradient for its weights, but these gradients will be added up
.. .. across each depth slice and only update a single set of weights per
.. .. slice.

.. .. kernel is another name for filter?

.. .. nice im2col explanation and conv demo, missing backprop example.

.. .. also expressing certain outputs with regular matmul with the
.. .. correct indexing is useful.

.. .. It is worth noting that there are only two commonly seen variations
.. .. of the max pooling layer found in practice: A pooling layer with
.. .. F=3,S=2F=3,S=2 (also called overlapping pooling), and more commonly
.. .. F=2,S=2F=2,S=2. Pooling sizes with larger receptive fields are too
.. .. destructive.

.. ..  In addition to max pooling, the pooling units can also perform
.. .. other functions, such as average pooling or even L2-norm
.. .. pooling. Average pooling was often used historically but has
.. .. recently fallen out of favor compared to the max pooling operation,
.. .. which has been shown to work better in practice.

.. .. Pooling: Notice that the volume depth is preserved.

.. .. backprop for max pooling is easy, got to keep track of where max
.. .. is. what to do if more than one max?

.. .. Recent developments.

.. Fractional Max-Pooling suggests a method for performing the pooling
.. operation with filters smaller than 2x2. This is done by randomly
.. generating pooling regions with a combination of 1x1, 1x2, 2x1 or 2x2
.. filters to tile the input activation map. The grids are generated
.. randomly on each forward pass, and at test time the predictions can be
.. averaged across several grids.
.. Striving for Simplicity: The All Convolutional Net proposes to discard
.. the pooling layer in favor of architecture that only consists of
.. repeated CONV layers. To reduce the size of the representation they
.. suggest using larger stride in CONV layer once in a while.
.. Due to the aggressive reduction in the size of the representation
.. (which is helpful only for smaller datasets to control overfitting),
.. the trend in the literature is towards discarding the pooling layer in
.. modern ConvNets.

.. It is worth noting that the only difference between FC and CONV
.. layers is that the neurons in the CONV layer are connected only to
.. a local region in the input, and that many of the neurons in a CONV
.. volume share parameters. 

.. fc->conv and conv->fc is interesting.
.. fc->conv has the advantage of using the whole net as a local filter
.. on a larger image!

.. Evaluating the original ConvNet (with FC layers) independently
.. across 224x224 crops of the 384x384 image in strides of 32 pixels
.. gives an identical result to forwarding the converted ConvNet one
.. time.

.. Another trick:
.. Lastly, what if we wanted to efficiently apply the original ConvNet
.. over the image but at a stride smaller than 32 pixels? We could
.. achieve this with multiple forward passes. For example, note that
.. if we wanted to use a stride of 16 pixels we could do so by
.. combining the volumes received by forwarding the converted ConvNet
.. twice: First over the original image and second over the image but
.. with the image shifted spatially by 16 pixels along both width and
.. height.

.. TODO: Inception module?

.. TODO: Batch normalization?

.. TODO: overfitting lecture: model size, early stop, good init,
.. regularization, bayes, dropout... need some theory.

.. TODO: optimization lecture: adam, rmsprop, adagrad... need some
.. theory.

.. TODO: knet and exercises.
