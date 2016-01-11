# Knet

.. _GitHub issue: https://github.com/denizyuret/Knet.jl/issues
.. _knet-users: https://groups.google.com/forum/#!forum/knet-users

Knet is a machine learning module implemented in Julia, so you should
be able to run it on any machine that can run Julia.  It has been
extensively tested on Linux machines with NVIDIA GPUs and CUDA
libraries, but most of it works on vanilla Linux and OSX machines as
well (currently cpu-only support for some operations is incomplete).
If you would like to try it on your own computer, please follow the
instructions on `Installation`_.  If you would like to try working
with a GPU and do not have access to one, take a look at `Using Amazon
AWS`_.  If you find a bug, or would like to request a feature, please
open a `GitHub issue`_.  If you need help please consider joining the
knet-users_ mailing list.

.. _fork the Knet repository: https://help.github.com/articles/fork-a-repo
.. _pull request: https://help.github.com/articles/using-pull-requests
.. _github.com: http://github.com

Knet is an open-source project and we are always open to new
contributions: bug fixes, new machine learning models and operators,
inspiring examples, benchmarking results are all welcome.  If you'd
like to contribute to the code base, please get an account at
github.com_ and `fork the Knet repository`_.  After your contribution
is implemented and passes ``Pkg.test("Knet")``, please submit it
using a `pull request`_.

