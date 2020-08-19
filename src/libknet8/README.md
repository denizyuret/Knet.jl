# Knet.LibKnet8: hand-written CUDA kernels.

The build.jl script can be used to build the libknet8.so shared library that includes
hand-written CUDA kernels. Knet automatically downloads a precompiled version using the
Artifacts package, so this is normally not necessary. The LibKnet8 module provides the
location of the libknet8 library and some macros for calling functions in that library.
