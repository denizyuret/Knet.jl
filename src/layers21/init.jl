export fanin, fanout, Uniform, U
using Base: to_shape, DimOrInd

struct Uniform; min; max; end
Uniform(x)=Uniform(-x,x)
(u::Uniform)(x...) = rand(x...) * (u.max-u.min) .+ u.min
const 𝑼 = Uniform

struct Normal; mean; std; end
Normal(x)=Normal(0,x)
(n::Normal)(x...) = randn(x...) * n.std .+ n.mean
const 𝑵 = Normal

fanin(dims::Dims{1}; o...) = dims[1]
fanout(dims::Dims{1}; o...) = 1
fanin(dims::Dims{2}; o...) = dims[2]
fanout(dims::Dims{2}; o...) = dims[1]

# if w is (3,3,16,8) with default layout (CUDNN_TENSOR_NCHW)
# or w is (16,3,3,8) with channelmajor (CUDNN_TENSOR_NHWC)
# then there are 16 input channels and 8 output channels
# fanin = 3*3*16 = (3*3*16*8) ÷ 8
# fanout = 3*3*8 = (3*3*16*8) ÷ 16
fanin(dims::Dims{N}; channelmajor=false)  where N = prod(dims) ÷ dims[N]
fanout(dims::Dims{N}; channelmajor=false) where N = prod(dims) ÷ (channelmajor ? dims[1] : dims[N-1])

# one can call fanin/out with arrays, sizes etc:
fanin(a; o...)=fanin(size(a); o...)
fanin(dims::Tuple; o...)=fanin(to_shape(dims); o...)
fanin(dims::DimOrInd...; o...)=fanin(to_shape(dims); o...)
fanout(a; o...)=fanout(size(a); o...)
fanout(dims::Tuple; o...)=fanout(to_shape(dims); o...)
fanout(dims::DimOrInd...; o...)=fanout(to_shape(dims); o...)
