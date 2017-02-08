.. Knet.jl documentation master file, created by
   sphinx-quickstart on Sat Oct 24 21:17:08 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Knet.jl's documentation!
===================================

Contents:

.. toctree::
   :maxdepth: 2

   install
   README
   backprop
   softmax
   mlp
   cnn
   rnn
   rl
   opt
   gen

.. kfun
.. nce
.. scode?
.. s2s
.. s2c
.. tagging
.. linreg
.. logistic/softmax
.. mlp
.. convnet
.. attention
.. perceptron
.. kernel perceptron

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. Here is some text.

.. .. Here is a reference :ref:`installation-section`

.. Here is some math: :math:`a^2+b^2=c^2`

.. Here is some Julia: ``square(x)=x*x``

.. Here is multiline Julia:

.. .. code::

.. 	function tagger_forw(net::Net, inputs...; o...)
.. 	    N = length(inputs[1])
.. 	    ystack = cell(N)
.. 	    for n=1:N
.. 	        ypred = forw(net, map(x->x[n], inputs)...; seq=true, o...)
.. 	        ystack[n] = copy(ypred)
.. 	    end
.. 	    return ystack
.. 	end

.. .. And that's that.
