## Installation

KUnet was tested on Julia v0.3 and v0.4 on Linux machines with and
without GPUs.  You should be able to install KUnet simply with:

```
julia> Pkg.clone("git@github.com:denizyuret/KUnet.jl.git")
julia> Pkg.build("KUnet")
```

This should automatically install other required packages
(e.g. InplaceOps, HDF5) if you don't have them already.  

There are also a number of optional packages KUnet can use if
installed: To work with a GPU (optional but highly recommended), you
need to have CUDA installed and add the following packages:

```
julia> Pkg.add("CUDArt")
julia> Pkg.add("CUBLAS")
```

Unfortunately CUBLAS has some incompatibilities with the latest
versions of Julia.  I recommend the following modifications in
CUBLAS.jl (which will be in ~/.julia/v0.4/CUBLAS/src after
installation):

```
typealias BlasChar Char #import Base.LinAlg.BlasChar
importall Base.LinAlg.BLAS
```
