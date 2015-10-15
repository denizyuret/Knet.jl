# Knet
Please see [A Compiler for Deep Learning](https://docs.google.com/document/d/1uyqKFEdCqS0HoA-acLUr9_C0EGmAx_0wiEwzXhMWUPw/edit?usp=sharing) for an introduction.  To install:
- Find a GPU machine.
- Setup your environment to see cuda and other necessary libraries.
- Install Julia from http://julialang.org/downloads: (v0.4.0) Generic linux binaries, 64-bit.
- Find and run the julia executable (usually in the bin subdirectory of the download).
- Install and build Knet with other required packages by typing the following at the julia prompt:
    - Pkg.init()
    - Pkg.clone("git://github.com/denizyuret/Knet.jl.git")
    - Pkg.checkout("Knet", "handout-mtg")
    - Pkg.build("Knet")
- To test the installation you can try some of the examples:
    - include(Pkg.dir("Knet/examples/adding.jl"))
- Please join the [knet-users mailing list](https://groups.google.com/forum/#!forum/knet-users) if you have any questions.
