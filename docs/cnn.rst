*****************************
Convolutional Neural Networks
*****************************


Motivation
----------

.. _ILSVRC: http://www.image-net.org/challenges/LSVRC/2014

Let's say we are trying to build a model that will detect cats in
photographs.  The average resolution of images in ILSVRC_ is
:math:`482\times 415`, with three channels (RGB) this makes the
typical input size :math:`482\times 415\times 3=600,090`.  Each hidden
unit connected to the input in a multilayer perceptron would have 600K
parameters, a single hidden layer of size 1000 would have 600 million
parameters.  Too many parameters cause two types of problems: (1)
today's GPUs have limited amount of memory (4G-12G) and large networks
fill them up quickly.  (2) models with a large number of parameters
are difficult to train without overfitting: we need a lot of data,
strong regularization, and/or a good initialization to learn with
large models.

One problem with the MLP is that it is fully connected: every hidden
unit is connected to every input pixel.  The model does not assume any
spatial relationships between pixels, in fact we can permute all the
pixels in an image and the performance of the MLP would be the same!
We could instead have an architecture where each hidden unit is
connected to a small patch of the image, say :math:`40\times 40`.
Each such locally connected hidden unit would have :math:`40\times
40\times 3=4800` parameters instead of 600K.  For the price (in
memory) of one fully connected hidden unit, we could have 125 of these
locally connected mini-hidden-units with receptive fields spread
around the image.

The second problem with the MLP is that it does not take advantage of
the symmetry in the problem: a cat in the lower right corner of the
image is going to be similar to a cat in the lower left corner.  This
means the local hidden units looking at these two patches can share
identical weights.  We can take one :math:`40\times 40` cat filter and
apply it to each :math:`40\times 40` patch in the image taking up only
4800 parameters.

.. TODO: add a picture of local vs fully connected.

A **convolutional neural network** (aka CNN or ConvNet) combines these
two ideas and uses operations that are local and that share weights.
CNNs commonly use three types of operations: convolution, pooling, and
normalization which we describe next.


Convolution
-----------

**Convolution in 1-D**

Let :math:`w, x` be two 1-D vectors with :math:`W, X` elements
respectively.  In our examples, we will assume x is the input
(consider it a 1-D image) and w is a filter with :math:`W<X`.  The 1-D
convolution operation :math:`y=w\ast x` results in a vector with
:math:`Y=X-W+1` elements defined as:

.. math::

   y_i \equiv \sum_{t=i}^{i+W-1} x_t w_{i+W-t} \,\,\forall i\in\{1,\ldots,X-W+1\}

This can be visualized as flipping w, sliding it over x, and at each
step writing their dot product into y.  Here is an example in Knet you
should be able to calculate by hand:

.. doctest::

   @knet function convtest1(x)
       w = par(init=reshape([1.0,2.0,3.0], (3,1,1,1)))
       y = conv(w, x)
       return y
   end
   julia> f = compile(:convtest1);
   julia> x = reshape([1.0:7.0...], (7,1,1,1))
   7x1x1x1 Array{Float64,4}: [1,2,3,4,5,6,7]
   julia> y = forw(f,x)
   5x1x1x1 Array{Float64,4}: [10,16,22,28,34]

.. _CUDNN: https://developer.nvidia.com/cudnn

``conv`` is the convolution operation in Knet (based on the CUDNN_
implementation).  For reasons that will become clear it works with 4-D
and 5-D arrays, so we reshape our 1-D input vectors by adding extra
singleton dimensions at the end.  The convolution of w=[1,2,3] and
x=[1,2,3,4,5,6,7] gives y=[10,16,22,28,34].  For example, the third
element of y, 22, can be obtained by reversing w to [3,2,1] and taking
its dot product starting with the third element of x, [3,4,5].


**Padding**

In the last example, the input x had 7 dimensions, the output y had 5.
In image processing applications we typically want to keep x and y the
same size.  For this purpose we can provide a ``padding`` keyword
argument to the ``conv`` operator.  If padding=k, x will be assumed
padded with k zeros on the left and right before the convolution,
e.g. padding=1 means treat x as [0 1 2 3 4 5 6 7 0].  The default
padding is 0.  For inputs in D-dimensions we can specify padding with
a D-tuple, e.g. ``padding=(1,2)`` for 2D, or a single number,
e.g. ``padding=1`` which is shorthand for ``padding=(1,1)``.  The
result will have :math:`Y=X+2P-W+1` elements where :math:`P` is the
padding size.  Therefore to preserve the size of x when W=3 we should
use padding=1.


.. doctest::

   @knet function convtest2(x)
       w = par(init=reshape([1.0,2.0,3.0], (3,1,1,1)))
       y = conv(w, x; padding=(1,0))
       return y
   end
   julia> f = compile(:convtest2);
   julia> y = forw(f,x)
   7x1x1x1 Array{Float64,4}: [4,10,16,22,28,34,32]

.. TODO: implement actual 1-D convolution.

For example, to calculate the first entry of y, take the dot product
of the inverted w, [3,2,1] with the first three elements of the padded
x, [0 1 2].  You can see that in order to preserve the input size,
:math:`Y=X`, given a filter size :math:`W`, the padding should be set
to :math:`P=(W-1)/2`.  This will work if W is odd.

**Stride**

In the preceding examples we shift the inverted w by one position
after each dot product.  In some cases you may want to skip two or
more positions.  The amount of skip is set by the ``stride`` keyword
argument of the ``conv`` operation (the default stride is 1).  In the
following example we set stride to W such that the consecutive filter
applications are non-overlapping:

.. doctest::

   @knet function convtest3(x)
       w = par(init=reshape([1.0,2.0,3.0], (3,1,1,1)))
       y = conv(w, x; padding=(1,0), stride=3)
       return y
   end
   julia> f = compile(:convtest3);
   julia> y = forw(f,x)
   3x1x1x1 Array{Float64,4}: [4,22,32]

Note that the output has the first, middle, and last values of the
previous example, i.e. every third value is kept and the rest are
skipped.  In general if stride=S and padding=P, the size of the output
will be:

.. math::

   Y = 1 + \left\lfloor\frac{X+2P-W}{S}\right\rfloor


.. TODO: mode is not very useful and is not supported by cpu, at some
.. point add it to the documentation.

**More Dimensions**

When the input x has multiple dimensions convolution is defined
similarly.  In particular the filter w has the same number of
dimensions but typically smaller size.  The convolution operation
flips w in each dimension and slides it over x, calculating the sum of
elementwise products at every step.  The formulas we have given above
relating the output size to the input and filter sizes, padding and
stride parameters apply independently for each dimension.

Knet supports 2D and 3D convolutions.  The inputs and the filters have
two extra dimensions at the end which means we use 4D and 5D arrays
for 2D and 3D convolutions.  Here is a 2D convolution example:

.. doctest::

   @knet function convtest4(x)
       w = par(init=reshape([1.0:4.0...], (2,2,1,1)))
       y = conv(w, x)
       return y
   end
   julia> f = compile(:convtest4);
   julia> x = reshape([1.0:9.0...], (3,3,1,1));
   julia> y = forw(f,x);
   julia> x
   3x3x1x1 Array{Float64,4}:
   [:, :, 1, 1] =
    1.0  4.0  7.0
    2.0  5.0  8.0
    3.0  6.0  9.0
   julia> get(f,:w)
   2x2x1x1 Array{Float64,4}:
   [:, :, 1, 1] =
    1.0  3.0
    2.0  4.0
   julia> y
   2x2x1x1 CudaArray{Float64,4}:
   [:, :, 1, 1] =
    23.0  53.0
    33.0  63.0

To see how this result comes about, note that when you flip w in both
dimensions you get::

   4 2
   3 1

Multiplying this elementwise with the upper left corner of x::

   1 4
   2 5

and adding the results gives you the first entry 23.

The padding and stride options work similarly in multiple dimensions
and can be specified as tuples: padding=(1,2) means a padding width of
1 along the first dimension and 2 along the second dimension for a 2D
convolution.  You can use padding=1 as a shorthand for padding=(1,1).

**Multiple filters**

So far we have been ignoring the extra dimensions at the end of our
convolution arrays.  Now we are ready to put them to use.  A
D-dimensional input image is typically represented as a D+1
dimensional array with dimensions:

.. math::

   [ X_1, \ldots, X_D, C ]

The first D dimensions :math:`X_1\ldots X_D` determine the spatial
extent of the image.  The last dimension :math:`C` is the number of
channels (aka slices, frames, maps, filters).  The definition and
number of channels is application dependent.  We use C=3 for RGB
images representing the intensity in three colors: red, green, and
blue.  For grayscale images we have a single channel, C=1.  If you
were developing a model for chess, we could have C=12, each channel
representing the locations of a different piece type.

In an actual CNN we do not typically hand-code the filters.  Instead
we tell the network: "here are 1000 randomly initialized filters, you
go ahead and turn them into patterns useful for my task."  This means
we usually work with banks of multiple filters simultaneously and GPUs
have optimized operations for such filter banks.  The dimensions of a
typical filter bank are:

.. math::

   [ W_1, \ldots, W_D, I, O ]

The first D dimensions :math:`W_1\ldots W_D` determine the spatial
extent of the filters.  The next dimension :math:`I` is the number of
input channels, i.e. the number of filters from the previous layer, or
the number of color channels of the input image.  The last dimension
:math:`O` is the number of output channels, i.e. the number of filters
in this layer.

If we take an input of size :math:`[X_1,\ldots, X_D,I]` and apply a
filter bank of size :math:`[W_1,\ldots,W_D,I,O]` using padding
:math:`[P_1,\ldots,P_D]` and stride :math:`[S_1,\ldots,S_D]` the
resulting array will have dimensions:

.. math::

   [ W_1, \ldots, W_D, I, O ] \ast [ X_1, \ldots, X_D, I ] 
   \Rightarrow [ Y_1, \ldots, Y_D, O ] \\

   \mbox{where } Y_i = 1 + \left\lfloor\frac{X_i+2P_i-W_i}{S_i}\right\rfloor

As an example let's start with an input image of :math:`256\times 256`
pixels and 3 RGB channels.  We'll first apply 25 filters of size
:math:`5\times 5` and padding=2, then 50 filters of size
:math:`3\times 3` and padding=1, and finally 75 filters of size
:math:`3\times 3` and padding=1.  Here are the dimensions we will get:

.. math::

   [ 256, 256, 3 ] \ast [ 5, 5, 3, 25 ] \Rightarrow [ 256, 256, 25 ] \\
   [ 256, 256, 25] \ast [ 3, 3, 25,50 ] \Rightarrow [ 256, 256, 50 ] \\
   [ 256, 256, 50] \ast [ 3, 3, 50,75 ] \Rightarrow [ 256, 256, 75 ]

Note that the number of input channels of the input data and the
filter bank always match.  In other words, a filter covers only a
small part of the spatial extent of the input but all of its channel
depth.

**Multiple instances**

In addition to processing multiple filters in parallel, we will want
to implement CNNs with minibatching, i.e. process multiple inputs in
parallel.  A minibatch of D-dimensional images is represented as a D+2
dimensional array:

.. math::

   [ X_1, \ldots, X_D, I, N ]

where I is the number of channels as before, and N is the number of
images in a minibatch.  The convolution implementation in Knet/CUDNN
use D+2 dimensional arrays for both images and filters.  We used 1 for
the extra dimensions in our first examples, in effect using a single
channel and a single image minibatch.  

If we apply a filter bank of size :math:`[W_1, \ldots, W_D, I, O]` to
the minibatch given above the output size would be:

.. math::

   [ W_1, \ldots, W_D, I, O ] \ast [ X_1, \ldots, X_D, I, N ] 
   \Rightarrow [ Y_1, \ldots, Y_D, O, N ] \\

   \mbox{where } Y_i = 1 + \left\lfloor\frac{X_i+2P_i-W_i}{S_i}\right\rfloor

If we used a minibatch size of 128 in the previous example with
:math:`256\times 256` images, the sizes would be:

.. math::

   [ 256, 256, 3, 128 ] \ast [ 5, 5, 3, 25 ] \Rightarrow [ 256, 256, 25, 128 ] \\
   [ 256, 256, 25, 128] \ast [ 3, 3, 25,50 ] \Rightarrow [ 256, 256, 50, 128 ] \\
   [ 256, 256, 50, 128] \ast [ 3, 3, 50,75 ] \Rightarrow [ 256, 256, 75, 128 ]

basically adding an extra dimension of 128 at the end of each data
array.  

By the way, the arrays in this particular example already exceed 5GB
of storage, so you would want to use a smaller minibatch size if you
had a K20 GPU with 4GB of RAM.

Note: All the dimensions given above are for column-major languages
like Knet.  CUDNN uses row-major notation, so all the dimensions
would be reversed, e.g. :math:`[N,I,X_D,\ldots,X_1]`.

**Backpropagation**

See http://people.csail.mit.edu/jvb/papers/cnn_tutorial.pdf for a
derivation of the backward pass for convolution.

.. TODO: summarize the derivative, maybe using 1D.

Pooling
-------

It is common practice to use pooling layers in between convolution
operations in CNNs.  Pooling reduces the size of its input by
replacing each patch of a given size with a single value, typically
the maximum or the average value in the patch.

Like convolution, pooling slides a small window of a given size over
the input optionally padded with zeros skipping stride pixels every
step.  By default there is no padding, the window size is 2, and
stride is equal to the window size.  The default pooling operation is
max.

**Pooling in 1-D**

Here is a 1-D example:

.. doctest::

   @knet function pooltest1(x)
       y = pool(x)
       return y
   end
   julia> f = compile(:pooltest1)
   julia> x = reshape([1.0:6.0...], (6,1,1,1))
   6x1x1x1 Array{Float64,4}: [1,2,3,4,5,6]
   julia> forw(f,x)
   3x1x1x1 CudaArray{Float64,4}: [2,4,6]

With window size and stride equal to 2, pooling considers the input
windows :math:`[1,2], [3,4], [5,6]` and picks the maximum in each
window.  

**Window**

The default and most commonly used window size is 2, however other
window sizes can be specified using the ``window`` keyword.  For
D-dimensional inputs the size can be specified using a D-tuple,
e.g. ``window=(2,3)`` for 2-D, or a single number, e.g. ``window=3``
which is shorthand for ``window=(3,3)`` in 2-D.  Here is an example
using a window size of 3 instead of the default 2:

.. doctest::

   @knet function pooltest2(x)
       y = pool(x; window=3)
       return y
   end
   julia> f = compile(:pooltest1)
   julia> x = reshape([1.0:6.0...], (6,1,1,1))
   6x1x1x1 Array{Float64,4}: [1,2,3,4,5,6]
   julia> forw(f,x)
   3x1x1x1 CudaArray{Float64,4}: [3,6]

With a window and stride of 3 (the stride is equal to window size by
default), pooling considers the input windows :math:`[1,2,3],[4,5,6]`,
and writes the maximum of each window to the output.  If the input
size is :math:`X`, and stride is equal to the window size :math:`W`,
the output will have :math:`Y=\lceil X/W\rceil` elements.

**Padding**

The amount of zero padding is specified using the ``padding`` keyword
argument just like convolution.  Padding is 0 by default.  For
D-dimensional inputs padding can be specified as a tuple such as
``padding=(1,2)``, or a single number ``padding=1`` which is shorthand
for ``padding=(1,1)`` in 2-D.  Here is a 1-D example:

.. doctest::

   @knet function pooltest3(x)
       y = pool(x; padding=(1,0))
       return y
   end
   julia> f = compile(:pooltest3)
   julia> x = reshape([1.0:6.0...], (6,1,1,1))
   6x1x1x1 Array{Float64,4}: [1,2,3,4,5,6]
   julia> forw(f,x)
   3x1x1x1 CudaArray{Float64,4}: [1,3,5,6]

In this example, window=stride=2 by default and the padding size is 1,
so the input is treated as :math:`[0,1,2,3,4,5,6,0]` and split into
windows of :math:`[0,1],[2,3],[4,5],[6,0]` and the maximum of each
window is written to the output.

With padding size :math:`P`, if the input size is :math:`X`, and
stride is equal to the window size :math:`W`, the output will have
:math:`Y=\lceil (X+2P)/W\rceil` elements.

**Stride**

The pooling stride is equal to the window size by default (as opposed
to the convolution case, where it is 1 by default).  This is most
common in practice but other strides can be specified using
tuples e.g. ``stride=(1,2)`` or numbers e.g. ``stride=1``.

.. TODO: fix infersize problem when stride != window.

In general, when we have an input of size :math:`X` and pool with
window size :math:`W`, padding :math:`P`, and stride :math:`S`, the
size of the output will be:

.. math::

   Y = 1 + \left\lceil\frac{X+2P-W}{S}\right\rceil

**Pooling operations**

There are three pooling operations defined by CUDNN used for
summarizing each window:

* ``CUDNN_POOLING_MAX``
* ``CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING``
* ``CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING``

These options can be specified as the value of the ``mode`` keyword
argument to the ``pool`` operation.  The default is
``CUDNN_POOLING_MAX`` which we have been using so far.  The last two
compute averages, and differ in whether to include or exclude the
padding zeros in these averages.  For example, with input
:math:`x=[1,2,3,4,5,6]`, ``window=stride=2``, and ``padding=1`` we
have the following outputs with the three options::

  mode=CUDNN_POOLING_MAX => [1,3,5,6]
  mode=CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING => [0.5, 2.5, 4.5, 3.0]
  mode=CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING => [1.0, 2.5, 4.5, 6.0]

**More Dimensions**

D-dimensional inputs are pooled with D-dimensional windows, the size
of each output dimension given by the 1-D formulas above.  Here is a
2-D example with default options, i.e. window=stride=(2,2),
padding=(0,0), mode=max::

   @knet function pooltest1(x)
       y = pool(x)
       return y
   end
   julia> f = compile(:pooltest1)
   julia> x = reshape([1.0:16.0...], (4,4,1,1))
   4x4x1x1 Array{Float64,4}:
   [:, :, 1, 1] =
    1.0  5.0   9.0  13.0
    2.0  6.0  10.0  14.0
    3.0  7.0  11.0  15.0
    4.0  8.0  12.0  16.0
   julia> forw(f,x)
   2x2x1x1 CudaArray{Float64,4}:
   [:, :, 1, 1] =
    6.0  14.0
    8.0  16.0


**Multiple channels and instances**

As we saw in convolution, each data array has two extra dimensions in
addition to the spatial dimensions: :math:`[ X_1, \ldots, X_D, I, N ]`
where :math:`I` is the number of channels and :math:`N` is the number
of instances in a minibatch.  

When the number of channels is greater than 1, the pooling operation
is performed independently on each channel, e.g. for each patch, the
maximum/average in each channel is computed independently and copied
to the output.  Here is an example with two channels::

  @knet function pooltest1(x)
      y = pool(x)
      return y
  end
  julia> f = compile(:pooltest1)
  julia> x = rand(4,4,2,1)
  4x4x2x1 Array{Float64,4}:
  [:, :, 1, 1] =
   0.0235776   0.470246  0.829754  0.164617
   0.375611    0.884792  0.561758  0.955467
   0.00740115  0.76617   0.674633  0.480402
   0.979588    0.949825  0.449385  0.956657
  [:, :, 2, 1] =
   0.254501  0.0930295  0.640946  0.270479
   0.422195  0.0399775  0.387326  0.234855
   0.102558  0.589408   0.69867   0.498438
   0.823076  0.797679   0.695289  0.888321
  julia> forw(f,x)
  2x2x2x1 CudaArray{Float64,4}:
  [:, :, 1, 1] =
   0.884792  0.955467
   0.979588  0.956657
  [:, :, 2, 1] =
   0.422195  0.640946
   0.823076  0.888321

When the number of instances is greater than 1, i.e. we are using
minibatches, the pooling operation similarly runs in parallel on all
the instances::

  julia> x = rand(4,4,1,2)
  4x4x1x2 Array{Float64,4}:
  [:, :, 1, 1] =
   0.664524  0.581233   0.949937  0.563411
   0.760211  0.714199   0.985956  0.478583
   0.190559  0.682141   0.43941   0.682127
   0.701371  0.0159724  0.28857   0.166187

  [:, :, 1, 2] =
   0.637187  0.279795  0.0336316  0.233479
   0.979812  0.910836  0.410312   0.94062 
   0.171724  0.388222  0.597548   0.817148
   0.41193   0.864101  0.178535   0.4956  

  julia> forw(f,x)
  2x2x1x2 CudaArray{Float64,4}:
  [:, :, 1, 1] =
   0.760211  0.985956
   0.701371  0.682127

  [:, :, 1, 2] =
   0.979812  0.94062 
   0.864101  0.817148


.. TODO: **Backpropagation**

Normalization
-------------

Draft...

Karpathy says: "Many types of normalization layers have been proposed
for use in ConvNet architectures, sometimes with the intentions of
implementing inhibition schemes observed in the biological
brain. However, these layers have recently fallen out of favor because
in practice their contribution has been shown to be minimal, if any."
(http://cs231n.github.io/convolutional-networks/#norm)  Batch
normalization may be an exception, as it is used in modern
architectures.

Here are some references for normalization operations:

Implementations:

* Alex Krizhevsky's cuda-convnet library API. (https://code.google.com/archive/p/cuda-convnet/wikis/LayerParams.wiki#Local_response_normalization_layer_(same_map))
* http://caffe.berkeleyvision.org/tutorial/layers.html
* http://lasagne.readthedocs.org/en/latest/modules/layers/normalization.html

Divisive normalisation (DivN):

* S. Lyu and E. Simoncelli. Nonlinear image representation
  using divisive normalization. In CVPR, pages 1–8, 2008.

Local contrast normalization (LCN):

* N. Pinto, D. D. Cox, and J. J. DiCarlo. Why is real-world visual
  object recognition hard? PLoS Computational Biology,
  4(1), 2008.
* Jarrett, Kevin, et al. "What is the best multi-stage architecture
  for object recognition?." Computer Vision, 2009 IEEE 12th
  International Conference
  on. IEEE, 2009. (http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf)

Local response normalization (LRN):

* Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
  classification with deep convolutional neural networks." Advances in
  neural information processing systems. 2012. 
  (http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2012_0534.pdf)

Batch Normalization:

* Ioffe, Sergey, and Christian Szegedy. "Batch normalization:
  Accelerating deep network training by reducing internal covariate
  shift." arXiv preprint arXiv:1502.03167 (2015). (http://arxiv.org/abs/1502.03167/)

.. TODO: LCN, LRN, DivN, BatchNormalization, Inception?


Architectures
-------------

TODO...

Exercises
---------

* Design a filter that shifts a given image one pixel to right.
* Design an image filter that has 0 output in regions of uniform
  color, but nonzero output at edges where the color changes.
* If your input consisted of two consecutive frames of video, how
  would you detect motion using convolution?
* Can you implement matrix multiplication in terms of convolution?
  reshape operations?  
* Can you implement convolution in terms of matrix multiplication?
* Can you implement elementwise broadcasting multiplication in terms
  of convolution?

References
----------

* Some of this was based on notes from: http://cs231n.github.io/convolutional-networks
* For derivatives see: http://people.csail.mit.edu/jvb/papers/cnn_tutorial.pdf
* The CUDNN manual has more details about the implementation: https://developer.nvidia.com/cudnn
* http://deeplearning.net/tutorial/lenet.html
* http://www.denizyuret.com/2014/04/on-emergence-of-visual-cortex-receptive.html
* http://neuralnetworksanddeeplearning.com/chap6.html
* http://www.deeplearningbook.org/contents/convnets.html
* http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution
* http://ufldl.stanford.edu/tutorial/supervised/Pooling
* http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork

.. TODO: mention the main motivation behind cnns, the visual cortex story.

.. TODO: separate programming examples from math?


.. TODO: add references at the end of each section.

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
