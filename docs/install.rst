.. _installation-section:

Installation
------------

First download and install the latest version of Julia from
`<http://julialang.org/downloads>`_.  Type ``julia`` at the command
prompt to start the Julia interpreter.  To install Knet type
``Pkg.clone("git://github.com/denizyuret/Knet.jl.git")`` and go get
some coffee while Julia downloads and installs all the necessary
packages::

    $ julia
                   _
       _       _ _(_)_     |  A fresh approach to technical computing
      (_)     | (_) (_)    |  Documentation: http://docs.julialang.org
       _ _   _| |_  __ _   |  Type "?help" for help.
      | | | | | | |/ _` |  |
      | | |_| | | | (_| |  |  Version 0.4.2 (2015-12-06 21:47 UTC)
     _/ |\__'_|_|_|\__'_|  |  Official http://julialang.org/ release
    |__/                   |  x86_64-apple-darwin13.4.0
    
    julia> Pkg.clone("git://github.com/denizyuret/Knet.jl.git")

If you have a GPU machine, you may need to type ``Pkg.build("Knet")``
to compile the Knet GPU kernels.  If you do not have a GPU machine,
you don't need ``Pkg.build`` but you may get some warnings indicating
the lack of GPU support.  Usually, these can be safely ignored.  To
make sure everything has installed correctly, type
``Pkg.test("Knet")`` which should take a couple of minutes kicking the
tires.  If all is OK, continue with the next section, if not you can
get help at the knet-users_ mailing list.

.. _knet-users: https://groups.google.com/forum/#!forum/knet-users

TODO:

* amazon machine
* github fork/pull
* github issues
