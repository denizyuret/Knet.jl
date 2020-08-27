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

fanin(dims::Dims{1}) = dims[1]
fanout(dims::Dims{1}) = 1
fanin(dims::Dims{2}) = dims[2]
fanout(dims::Dims{2}) = dims[1]
# if a is (3,3,16,8), then there are 16 input channels and 8 output channels
# fanin = 3*3*16 = (3*3*16*8) ÷ 8
# fanout = 3*3*8 = (3*3*16*8) ÷ 16
fanin(dims::Dims{N}) where N = prod(dims) ÷ dims[N]
fanout(dims::Dims{N}) where N = prod(dims) ÷ dims[N-1]
# one can call fanin/out with arrays, sizes etc:
fanin(a)=fanin(size(a))
fanin(dims::Tuple)=fanin(to_shape(dims))
fanin(dims::DimOrInd...)=fanin(to_shape(dims))
fanout(a)=fanout(size(a))
fanout(dims::Tuple)=fanout(to_shape(dims))
fanout(dims::DimOrInd...)=fanout(to_shape(dims))
