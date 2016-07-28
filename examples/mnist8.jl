using Knet
isdefined(:MNIST) || include("mnist.jl")

# 1. Here is the simple softmax model:

w = convert(Array{Float32},0.1*randn(10,784)) # TODO: randn should take Float32 like rand
b = zeros(Float32,10)
soft8(x)=Knet.softforw(x,x)
x = reshape(MNIST.xtst, (784,10000)) # '*' should accept 4D arrays
y = soft8(w*x.+b)

# 2. Now try with special types:

abstract Arr8                   # internal array
type Par8 <: Arr8; a; end       # internal parameter array
type Dat8 <: Arr8; a; end       # internal data array

# If any argument is Arr8, the output is a Dat8.

importall Base

for op in (:*, :.+)
    @eval begin
        $op(a::Arr8,b::Arr8)=Dat8($op(a.a,b.a))
        $op(a::Arr8,x)=Dat8($op(a.a,x))
        $op(x,a::Arr8)=Dat8($op(x,a.a))
    end
end

soft8(a::Arr8)=Dat8(soft8(a.a))

w8 = Par8(w)
b8 = Par8(b)
y8 = soft8(w8*x.+b8)

# 3. operations with parameter arrays and anything they touch (call
# these internal arrays) need to be recorded

# Recording problems:
# - arrays not changed need not be copied
# - arrays not needed for gradient need not be copied
# - support for both inplace and allocating ops
# - need identity (for cumulative gradient) and current contents of the arrays
# - cumulative vs one-shot gradients?
# - arrays that get overwritten?
# - can assume parameters not overwritten during a session.
# - simple copy works for sparse, cuda etc. no need to deepcopy a.a
# - need op, out, ins, both id & contents, also gradient arrays
# - some of out, ins could be regular arrays, Par, or Dat.
# - Par never change, no need to copy, gradient always cumulative.
# - Dat and Array can be overwritten, or be used multiple times.
# - We just need to treat overwriting as just a new array or ban it.
# - That means id of the same array may be different after overwriting?

# TODO:
# - Record
# - Loss
# - Backprop
# - InplaceOps
# - Optimization
