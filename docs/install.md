## Installation

KUnet was tested on Julia v0.3 and v0.4 on Linux machines with and
without GPUs.  You should be able to install KUnet simply with:

```
julia> Pkg.clone("git@github.com:denizyuret/KUnet.jl.git")
```

This should automatically install other required packages
(e.g. InplaceOps) if you don't have them already.  There are also a
number of optional packages KUnet can use if installed:

To work with a GPU (optional but highly recommended), you need to have
CUDA installed and add the following packages:

```
julia> Pkg.add("CUDArt")
julia> Pkg.add("CUBLAS")
```

KUnet supports saving and loading networks in HDF5 format if you
install this optional package:

```
julia> Pkg.add("HDF5")
```

