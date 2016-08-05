# Main idea of Knet8: let the user write and run an ordinary Julia
# program for the forward computation.  Extra steps: (1) certain
# variables designated as parameters, (2) certain results are paired
# with gold values and a loss function, (3) automatic gradient and
# updating.

# Motivation: be able to use all control operations of Julia without
# worrying about compiling them.

using Knet
importall Base
isdefined(:MNIST) || include("mnist.jl")

### 1. Here is the simple linear model:

w1 = convert(Array{Float32},0.1*randn(10,784)) # TODO: randn should take Float32 like rand
b1 = zeros(Float32,10)
x1 = reshape(MNIST.xtst, (784,10000)) # TODO: '*' should accept 4D arrays
y1 = w1*x1.+b1
display(y1)
@show quadloss(y1, MNIST.ytst)

### 2. Now try with special types:

# abstract Arr8                   # internal array
# type Par8 <: Arr8; a; end       # internal parameter array
# type Dat8 <: Arr8; a; end       # internal data array

# # If any argument is Arr8, the output is a Dat8.

# for op in (:*, :.+)
#     @eval begin
#         $op(a::Arr8,b::Arr8)=Dat8($op(a.a,b.a))
#         $op(a::Arr8,x)=Dat8($op(a.a,x))
#         $op(x,a::Arr8)=Dat8($op(x,a.a))
#     end
# end

# w2 = Par8(w1)
# b2 = Par8(b1)
# x2 = x1
# y2 = w2*x2.+b2
# display(y2)

# TODO: do a full register, define, minibatch, train, etc. from the
# users perspective to make sure everything works.  Then worry about
# memory efficiency.

### 3. operations with parameter arrays and anything they touch (call
# # these internal arrays) need to be recorded

# type Rec8; op; out; args; end
# Rec8(op,out,args...)=Rec8(op,out,args)

# ST8 = Rec8[]

# function call8(f, args...)
#     out=Dat8(f(args...))
#     push!(ST8, Rec8(f,out,args))
#     return out
# end

# for op in (:*, :.+)
#     @eval begin
#         $op(a::Arr8,b::Arr8)=call8($op,a.a,b.a) # TODO: lost the identity of a & b, recording arrays instead? where are the gradients going to be?
#         $op(a::Arr8,x)=call8($op,a.a,x)         # TODO: ignoring possible overwrites.
#         $op(x,a::Arr8)=call8($op,x,a.a)
#     end
# end

# w3 = Par8(w1)
# b3 = Par8(b1)
# x3 = x1
# y3 = w3*x3.+b3
# display(y3)


### 4. gold output needs to be recorded
# # during function call?
# # after function call, loss(ypred,ygold)?

# function want8(ypred::Arr8, ygold; loss=softloss)
#     # do we return loss here?  loss+grad leads to inefficiency...
#     push!(ST8, Rec8(loss, ypred, ygold))
# end

# ygold = MNIST.ytst
# want8(y3, ygold)                # TODO: wrong loss function without softmax

# # gradients:
# # should we combine gradients and back?
# # need a gradient field in Arr8
# # where do we store the back functions?  gradient registration?  function or method based?
# # In general think carefully about extending the system (by registering more gradients and optimization methods?)

### 5. new design:
# a. combine Par, Dat, and Rec into Reg.
# b. solve registration of ops and loss functions.

# """
# Reg8: DataType for marking parameters and descendents and recording operations.
# For now we'll assume Reg8.out and arrays in Reg8.args do not get overwritten.
# op: operation that created the variable.  This is typically a function.  For parameters and constants it may have a dummy value.
# args: inputs to the operation.  Only pointers are kept (not copies).
# out: the output of the operation.
# dif: gradient array, same size/type with out.
# """
# type Reg8; out; op; args; dif; end

# function writemime(io::IO, ::MIME"text/plain", r::Reg8)
#     @printf("%s,%s,%s",r.op,summary(r.out),join(map(summary,r.args),","))
# end

# # St8: global stack
# isdefined(:St8) || (St8 = Reg8[])

# "Par8: constructor for parameters."
# Par8(out)=push!(St8, Reg8(out, Par8, [], nothing))[end] # TODO: do we need to record parameters?

# # function registration example: TODO: turn this into a macro.

# for op in (:*, :.+)
#     @eval function $op(args...)
#         regs = false
#         getout(x)=(isa(x,Reg8) ? (regs=true; x.out) : x)
#         outs = map(getout, args)
#         regs || die("No registers found.")
#         out = $op(outs...)
#         reg = Reg8(out, $op, args, nothing)
#         push!(St8, reg)
#         reg
#     end
# end

# w5 = Par8(w1)
# b5 = Par8(b1)
# x5 = x1
# y5 = w5*x5.+b5
# display(y5)

# # TF implements loss calculation as part of the TF graph
# # Then it calls minimize instead of loss
# # Can we do the same?
# # We allow for loops etc but not within arrays!  i.e. kernels that modify arrays should be defined outside of Knet8.
# # Still, a loss function can be like any other operation?
# # But what if there are multiple losses like in a sequence model, need to add the losses and minimize the result.
# # Can we have scalar Knet variables, add them, and minimize the result?
# # loss forw can compute the loss value, back can compute the loss gradient.
# # loss calls also mark the places in comp graph the values we are trying to minimize. (we always minimize the sum)


# # Recording problems:
# # - arrays not changed need not be copied
# # - arrays not needed for gradient need not be copied
# # - support for both inplace and allocating ops
# # - need identity (for cumulative gradient) and current contents of the arrays
# # - cumulative vs one-shot gradients?
# # - arrays that get overwritten?
# # - can assume parameters not overwritten during a session.
# # - simple copy works for sparse, cuda etc. no need to deepcopy a.a
# # - need op, out, ins, both id & contents, also gradient arrays
# # - some of out, ins could be regular arrays, Par, or Dat.
# # - Par never change, no need to copy, gradient always cumulative.
# # - Dat and Array can be overwritten, or be used multiple times.
# # - We just need to treat overwriting as just a new array or ban it.
# # - That means id of the same array may be different after overwriting?
# # - There is also reshape we need to think about.
# # - Will we ever need multiple sessions?
# # - 3-place vs 2-place operators, memory management optimization?
# # - Function registration: a .= b*c calls A_mul_B!, do we have to register both? how do we register gradients?
# # TODO:
# # - Record
# # - Loss
# # - Backprop
# # - InplaceOps: julia 0.5 has built-in ones!
# # - Optimization


### 6. Autograd style: params=args, inputs=global, return=loss

function model6(w)
    x = reshape(MNIST.xtst, (784,10000))
    ygold = MNIST.ytst
    ypred = w[1]*x .+ w[2]
    quadloss(ypred, ygold)
end

@show model6((w1,b1))

# OK, now we have to make this work:
# grad_model6 = grad(model6)
# grad_weight = grad_model6((w1,b1))

