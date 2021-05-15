## Knet.CUDNN: High level interface to cuDNN functions

The goal of this submodule is to make the high level CUDA.CUDNN interface work for
KnetArrays and to define gradients. See CUDA.CUDNN README for design choices.

To test CUDA.CUDNN go to the CUDA root directory and use:

    julia --project -e 'using Pkg; Pkg.API.test(; test_args=`--sanitize --jobs=5 cudnn`)'
