## Installation

KUnet was tested on Julia v0.3 on Linux machines with and
without GPUs.  You should be able to install KUnet simply with:

```
julia> Pkg.clone("git://github.com/denizyuret/KUnet.jl.git")
julia> Pkg.build("KUnet")
```

This should automatically install other required packages
(e.g. Compat, HDF5) if you don't have them already.  

I left the GPU packages optional to allow installation on non-GPU machines.
To work with a GPU (optional but highly recommended), you
need to have CUDA installed and install the following packages:

```
julia> Pkg.add("CUDArt")
julia> Pkg.add("CUBLAS")
julia> Pkg.add("CUDNN")
```
