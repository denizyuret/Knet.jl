***********************
A Tutorial Introduction
***********************

TODO:

- we need to talk about installation somewhere.
- Other requirements like v0.4.0, cuda libraries, cpu compatibility etc.
- + Install latest v0.4.2.
- + Update packages.
- + Figure out no-gpu installation (CUDA* requirements)
- Create an amazon aws image for easy gpu work.

We will begin by a quick tutorial on Knet.

.. see http://sphinx-doc.org/ext/doctest.html
.. testcode for regular doctest for prompted examples
.. http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html#directives

.. testcode

    @knet function linreg(x)

.. testcode::

    a = 1 + 2

.. testcode::

    a + 4

.. testoutput::

    7


- kfun as model: linear regression.
- kfun as new ops: mnist lenet.
- compile time parameters: 
- runtime parameters: conditionals: dropout? on mnist lenet?
- rbw registers: rnn intro, rnnlm (char based).
- conditionals: copyseq or adding or dropout?

- linear regression?  uci?  https://archive.ics.uci.edu/ml/datasets/Housing
- or do we do artificial data generation: cpu/gpu conversion may be difficult.
- mnist definitely
- mnist4d for convolution
- maybe something else for simple nnet?
- copyseq to introduce rnns
