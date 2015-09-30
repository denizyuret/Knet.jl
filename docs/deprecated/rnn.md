rnn consists of an array of layers.
each layer keeps an array of outputs?
dw accumulated going back.
each layer has an input.
rnn[i] may have an input rnn[j<i][t=0] or rnn[j][t>0]
which means rnn[j] needs to keep its output for all [t]
what happens when t is invalid?  (we are at T=1 and t=3 for some input)?
adder layers need more than one input.

rnn-forw:
start at rnn[i=1], loop rnn[i+=1]
how do we give initial inputs?
find out where to get x for each layer and call its forw with that x
how do we collect final outputs?

rnn-back:
we have to give x as well as dy
we find the x just like forw
which means each layer is responsible for its own output history
layers should not keep their own x?
should they manage their own y?
state-less layers?  forw computes y given x, nothing kept.  preallocated y can be passed.  or x overwritten.
can we allow any overwriting in rnn?  what if we need the x before overwriting?  do we ever?
back takes dy and x giving dx and accumulating dw.
who zeros dw?
who resets history?
do we have a time variable?
rnn-forw could pass in time.

what about add layers?

mathematically the rnn order does not matter.
so we actually have a set of nodes.
each node (except adder) points to a single parent node with a possible time delay.
multiple nodes can point to the same parent node (get their inputs from it) with identical or different delays.

what happens to overwriting layers?

so we could give all responsibility to nodes rather than rnn-forw.
they know their input.
rnn order does matter for serial computation order respecting dependencies.
rnn-forw can call layer-forw with a single time parameter.
input node can have all its ylist filled in the beginning.

or nodes don't keep anything, rnn-forw calls them with x and y,
rnn-back calls them with dy, x, and dx (do we need both x and dx? can
dx overwrite x? can dy and y share storage?
storage (other than parameters) is managed outside of the layers.
layers implemented as stateless functions.

dw accumulation?  resetting?  initback?

OPTION 1: rnn-forw manages storage, calls layers with pre-allocated x/y.
OPTION 2: layers manage storage, have history arrays, rnn-forw calls them with time.

OPTION 1 sounds better for now.

We should think of this top-down.

We have some sequence data: (x1,y1), (x2,y2), ...
where xi and yi are sequences of tokens (represented by one-hot or other vectors).

Shall we represent xi as a matrix, or as a vector of vectors?

How will batching work?

We process multiple xi together.  And propogate errors compared to yi.

What if xi's are of different length?  We can't resize the input matrix like we do 
because the back phase does not start until all sequences in the batch are done.

This is similar to the situation in beam parser.

We can shrink the input batch as we move forward?

Let's replicate marjan's reversing experiment.

Claim: no architecture ever looks back more than one time step in forw.
Claim: you only look back if your input index is >= your index.
This will eliminate the time coordinate from the network description.
We still need the Add2 layer.
We can do the binary adder example.
or a language model.  both have well specified input/output at each step.

Classification problems have a bunch of dont-cares followed by one output.
Also the addition example from irnn paper.

Sequence mapping models have an encoding rnn and a decoding rnn
trained together.  Need special code for that.

There is no difference in the forward algorithm.  We just feed in a
sequence of x.  We evaluate using a subset of y.  We may need beam
search for evaluation.

Here is the adder:

[(Mmul(h), 0), 	# 1 takes input x
 (Mmul(h), 5), 	# 2 takes prev hidden state
 (Add2(), 1, 2), # 3 adds together
 (Bias(), 3),	 # 4 specify bias separately
 (Relu(), 4),  	# 5 this is the next hidden state
 (Mmul(y), 5),	# 6 computing output
 (Bias(), 6),	# 7 computing output
 (XentLoss(), 7),	# 8 last layer is output
]

Compare this to a regular single hidden layer feed forward net:
[Mmul(h), Bias(), Relu(), Mmul(y), Bias(), XentLoss()]

In fact if we make the default input the previous one or two layers
(keeping backward compatible) we only need to specify it when we have
to:

[Mmul(nh), (Mmul(h),5), Add2(), Bias(), Relu(), Mmul(ny), Bias(), XentLoss()]

OK: Barrett says he gets rid of the need to train an initial hidden
state by always feeding in the same start symbol.  The embedding of
the start symbol gets trained to give the initial hidden state.  This
means the input and output of the hidden-to-hidden unit are both 0.
We can adjust Add2 so if one of its inputs is nothing it just gives
the other one.  Same with Mul2 giving out zero.


### Can we have a single Net type?

What distinguishes rnn from fnn?  
Both can be defined as a sequence of layers with associated input specs.
; Inputs should probably be defined with relative coordinates for easier cut-paste.
The default input spec is the previous layer(s) in sequence
; but even fnn's may require input specs if they have skip connections.

Difference 1: rnn's have recurrent connections (inputs from forw layers).
a. this makes no difference in forw calculation, both just keep track of the last output of each layer.
; this means rnn and fnn can have the same predict function.
; but not the same forw function, if rnn needs to remember stuff.
b. rnn's have to keep stuff around for bptt calculation, fnn's don't?
;; do they really?  or is this an efficiency issue only?  
; fnn's can also do updates at the end of the sequence?
;; trivially yes, but at the expense of extra storage.
; rnn's can also do updates every time step?
;;; back calculations are additive.  
;;; if your output goes to multiple places, their contribution to your dy adds up.
;;; instead of adding dy=dy1+dy2 and going back, can we do back(dy1) followed by back(dy2)?
;;;; this comes down to whether back is a linear function of dy:
; mmul, bias, mul2, add2: yes
; conv, pool: ?
; loss: we won't have multiple dy's coming into this one.
; actf: yes
;; sigm: dx = dy * y * (1-y)
;; tanh: dx = dy * (1+y) * (1-y)
;; relu: dx = dy * (y>0)
;;; so the answer seems yes, back(dy1+dy2)=back(dy1);back(dy2)
;;; even if the answer is yes, some layers require x/y for back calculation
;;; in rnn there is a contribution from y/dy at t=T all the way back to layers at t=1
;;; and this contribution requires all the intermediate x's to be kept even if we were doing back at each step.

; conclusion: yes, rnn's can do back every time step
;; however they still need to keep around x's until the end of the sequence.
;; and it would be less efficient then doing back(dy1+dy2), tracing every back path once.

; So even if we had the same Net type for rnn's and fnn's:
;; we would have to detect when a net is an rnn
;; we would have to use different forw functions for rnn-test vs rnn-train.

Difference 2: rnn's get fed sequence inputs.
; fnn's can also be fed sequence inputs: successive inputs do not effect the next time step.
; rnn's can also be fed non-sequence inputs: each input can be assumed the next in sequence, until a special end-of-sequence input (maybe nothing?).

Difference 3: rnn's wait until the end of the sequence before "back".

; important thing is dw, dy/dx is only necessary they support dw calculation.
; only bias, conv, and mmul have dw calculation.
;; any back path that does not lead to a bias/conv/mmul can be stopped.
;; any forw path that does not lead to an output can be stopped.

### Can we have reversible RNNs?

;;;; can the x's be recalculated using a reverse forw function?
;;;; no, add2 and mul2 lose information 
; although we could have two outputs with two inputs and make them reversible
; however this will not prevent having to remember something for each time step of forw?
;;;; mmul loses information unless w is square.
;;;; bias is reversible.
;;;; tanh/sigm reversible, relu loses information.
;;;; i think it is worth thinking about reversible rnn's.

### Input format:

with rnn's we have the option to present forw with:
1. contiguous input (xdim, minibatch, time)
2. vs array input (vector of (xdim, minibatch))
since I am not keeping the other layer outputs contiguous #2 seems better.
barrett claimed #1 is more efficient but my copy! tests dont show much difference.
; if we consolidate rnn and fnn types we may need to have two different forw functions
; that interpret the input as a sequence of minibatches vs as a minibatch.
;; we could determine that from the type, i.e. Vector of Arrays vs a single Array.
; the rnn forw function further splits into predict vs train mode.

we dont want a low level forw function that takes the whole sequence?
it would take less memory to feed in each time step.
in other words we should take the t=1:T for loop outside the forw.
in predict mode, don't need to set/read r.h, so no need for time index.
in train mode we could keep an internal timer.
however we do need to keep input x[1:T] for back calculations.  
so preserve using r.x, this also solves the problem of layers that overwrite their input.
and forw/back never take sequences!
back can simply accept dy=nothing for some time steps?
we need to feed x forwards and dy backwards in train.
no need to use subarrays for r.h.
but we need to reset t=0 at the start of each sequence
; actually that will automatically happen if back subtracts from r.t
should we still keep r.x separate from r.h?


### Can we have a recursive Net type?

; need to consolidate the types RNN, Net, and Layer.
; do we append or list when merging?
; say we use list

Have an abstract type Net.  It supports:
forw(nn,x)->y
back(nn,dy)->dx

We can put multiple nn's together, the resulting thing should also be an Net.
We should be able to use it as a primitive.

As soon as we have recurrent connections, we need to keep history.

This should work for all r::Net:
for n=1:N; r.y[n] = forw(r.f[n], r.y[r.inputs[n]]...; o...); end

Except primitive ops, which just implement their own forw.
; which will just work, due to type overriding.
Except when reset (x=nothing?) when we should set all r.y[:]=nothing.
; that is easy to implement
; for primitives we used nothing as zero matrix so not a good idea.
; have a reset function instead for rnns.
Except skip input ffnn's which have to be careful with overwriting layers.
; when freads > 1 and the first child overwrites, second will have the wrong input.
Except rnn's which have to store some history for back.
; does this work for embedded RNNs?

A Layer is a Net.
A RNN is a Net.
RNN consists of a list of Nets, and their input specs.

Net : RNN | Layer
RNN : List<Net>, Inputs

We don't need three types, we only need two:

Layer : Net | Mmul | ...
Net   : List<Layer>, Inputs

Layer is a bad name, Atom? Func? Node? Op? Fn? Primitive?
Caffe calls them Layer, Theano calls them Op.  Go with Op:

### Recursive Net type, Try 2:

Op : Net | Mmul | ...
Net: List<Op>, Inputs

Each Op supports:
forw(op,x)  -> y
back(op,dy) -> dx

Storage for y and dx:
- Either x and dy are overwritten
- Or internal storage resized and used (in which case it is overwritten next call)
- overwrites(op) tells us which.
- overwrites(net) has to figure it out from its components.
- in case of net, both overwriting and internal storage may be used, PrimOps use one or the other.
- PrimOps always overwrite dy? or only when x is overwritten?
;; seems the second: mmul,mul2,conv,pool allocate; bias,add2,actf,loss,drop overwrite (or pass without overwriting, which should be marked as overwrite)
;; when is dy used more than once?  only when there is more than one input: mul2 and add2.
;; so it is ok to assume dy always overwritten.
- overwrites(net) is true if any of its ops with input=0 overwrites.
- supporting multi-input Ops is tricky, add2 only overwrites its second argument.
;; we could just not overwrite, but it also passes dy back twice?  we pass dy and a copy of dy.
;; mul2 allocates dx1 and dx2, we could similarly use dy for one of them.

Overwriting is important in forw(net)
- if an r.y[i] is going to be overwritten but it is going to be used again, it needs to be copied to r.y[n].
- so if we just don't overwrite in add2, all will be simplified and forw is done.

Back:
- in ffnn's and PrimOps there is a forw and back after every input.
- in rnn's there is multiple forw followed by an equal number of back.
- we could also run multiple forw in ffnn's as long as we remember stuff for back.
- can we make PrimOps smarter to keep multiple x's or y's for training?
;; PrimOps that support multiple forw followed by multiple backs: that way Net does not have to remember anything.
;; there is two types of forw: predict and train.
;; for predict we don't have to remember anything, there is no back.
;; for train each Op can keep track of the stuff it needs for back.
;; should we copy or point?  always copy, everything can be overwritten?
;; each PrimOp that keeps state needs a timer t.  forw(train) increments, back decrements 
;; forw(predict) does not touch timer t (do we need reset?)
;; back_reads_x: conv, pool, mmul, mul2
;; back_reads_y: actf, loss, pool

Can we really simplify RNNs?
forw simply calls forw of each op.
back simply calls back of each op in reverse.
everybody keeps track of t.
we don't need back_reads_x and back_reads_y.
do we have to have overwrites? can't we assume we always overwrite?
we could copy any x with multiple forwreads.

Can we use this for non-RNNs?
t would only be 1.
layers would store what they will need.
any unnecessary copying?
have to store y -> others may write on it.
possible extra storage of x -> if interm layer it was going to get overwritten, if first layer it is the original input, somebody else may read and overwrite.

Problem:
same input will be stored multiple times by different Ops.
Example: 
- lstm needs x, h for mmul.x for 4 different mmul pairs.
- it needs i, f, o, c1, c, c2 for mul2.x.
- it needs i, f, o, c1, c2 for actf.y.
So the union is 8 cells.  If each unit copies we have 19.  We have shared references.

Can we keep reference counts?  Ops never copy, they always point, the increment reference count.
We don't overwrite if refcount > 0.
However this does not result in reuse?
It also does not solve the forw problem, i.e. future forw reads.

Forget it, Net has to keep track of stuff for back and present it to back(op) with specific x,y.
In which case we are using back(op) as a function, i.e. all inputs specified. Though output still allocated.

Since cells can be shared among ops, we can't rely on ops to do memory management.


### Recursive Net type, Try 3:

Op: Mmul | ...
Net: List<Op>, Inputs

Each Op supports:
forw(op,x,y)   -> y
back(op,dy,dx) -> dx

Ops do not allocate outputs or keep record of past inputs, they are memoryless functions.

However if Net is an Op and it needs memory for RNN!

What happens when a Net has a member that is a Net?  Who keeps the memory?

This argues for Nets not to be Ops, when a Net is a member of another, it should be spliced by the constructor.

There should be only one memory manager: implemented inside Net.

We can't have embedded Nets.  Nets cannot be Ops.  However they can be passed to constructors for other nets, which should splice them, so the resulting net is a simple sequence again.

Call them OP and NN?

Net constructor is basically a compiler.

So we are back to having r.y, r.Y, r.dy, etc.  Except the constructor should accept other nets and splice them.

The basic constructor should just do the splicing, fixing the indices.

Compiler should do memory allocation after receiving the first input (initforw)?

r.y should have the minimum number of required arrays, with sharing where possible.

r.y[n] should be passed as an option to forw.  can be shared where appropriate.

Remove all memory management from layers?

r.dy[i] will have to be allocated as well (initback), again with sharing where possible.

We will still keep stepwise forw and stepwise back.

NO: We need allocy(op,x) and forw(op,x;y=allocy(op,x)) or forw(op,x;y=x) supported for each op.  
;; no need if we use resize
NO: forget about keeping and reusing l.y.
;; We can still have y=nothing as a keyword arg and use y=x or y=l.y if not specified.

We won't know the sizes unless we do a trial run, which we can after the first x.
OK: Or we can just assign size-0 KUdense everywhere and have layers resize as appropriate?
OK: We also need to resize appropriately?  should rnn do this or the ops? ops can do it.
In which case we don't need allocy except internally to an Op.  

TODO: renames after testing:
RENAME: KUdense -> KUarray
DONE: Net -> MLP
DONE: RNN -> Net
DONE: Layer -> Op
DONE: predict=false -> train=true
DONE: RNN2 -> S2C

During construction and compilation we will know N but we wont know T.
So things with T dimensions: r.x, r.h need to be dynamically resized.
We can use KUarrays for those as well.

As we are growing r.h adding another column, we need to keep the same
pattern of shared arrays from the previous column.
rnn only takes care of sharing: we can keep track of yarray[n] and harray[n] indices.
or we can keep an example h to be concatenated.

r.t only kept track of by Net.
during train forw increments, back decrements.
during predict forw increments (so we know if we are in the beginning and don't use r.y from previous sentence).

what about r.x, do we need it?  do we copy?  
treat it like another Op.
since we are modifying the list during Net construction why not add a data Op.
no input, (just like Rand ops), x output: point or copy?
just like other layers.
look at forwreads and backreads.
0 means no input, 1 means data input.
we'll have to perform index transformations anyway.

Net constructor: how to splice:
Initialize the net with the input layer (i.e. op=nothing, inputs=(0,))
If there is another net member remember it also has an empty input layer.
we need an index translation table.
what the user meant by j was the output of the j'th element of his list.
what the user meant by 0 was the input to the Net.
we need to turn j into the output index of the j'th element.
j'th element can be an Op or a Net.
j could be in the future.
can we first compute the translation table?

# allocate optimize output registers
# allocate optimize history registers
The following should be the code for forw:
r.output[n] = forw(r.op[n], r.inputs[n]...; y=r.output[n], o...)
But there is the issue of back pointers at t=1, represented by "nothing" inputs
We may have to keep the netinputs function.
(need two names, one for index transformation one for actual input gathering.)

Assumption: A Net cannot have multiple inputs.
In that case we cannot express LSTM easily, it has four multipliers.
Think again...
New design: inputs indicated by 0, -1 etc.
We use the wrap around convention, i.e. 0 means end-0, -1 means end-1
;; Alternatively just add to the end of list: 0 means n+1+0, -1 means n+1+1 etc.
;; when we concatenate the indices 0, -1 etc. point to the last few outputs
;; add2 and mul2 commutative so I guess it doesn't matter which convention for now.
so we can store the inputs at the end of h and y
and we can keep the convention y[n] is the output of op[n].
and we can get rid of the Data() layer.
We allow Net's to have multiple inputs as well.
Can we have the following convention:
- The first Op in a Net determines the number of inputs: ?
- The last Op in a Net determines the output: OK.

naming problems:
opin
opout
netin
netout
index vs array

We have problems with the nothings.


# Implement back
# test mnist

# back: state of r when back is called:
# time = T
# buffer[n] = last output of op[n]
# stack[n][1..T] = outputs of op[n] for t=1..T
# back gets dy (desired value of output[N])
# for n=N:-1:1
## find output and inputs of op[n] at time T
## some may be from T-1, look these up from stack

## stack for inputs separated?
## no stack read/write for direct forw/back?
## inputs and output contain register ids?
## build things around get/set registers?
## have a set of registers that is a subset of ops
## make some of these registers keep history
## use push/pop?
## do we ever need access to reg[n][t] and reg[n][t-1]?
# op[5][t] has inputs from op[3][t] and op[7][t-1]
# we are going back so op[7][t] has been processed, it is ok to pop
# push(r,n) will save the current value of reg[n] on stack
# pop(r,n) will pop the previous value of reg[n] from stack
# reg(r,n) returns the current contents of register n
# we never need to access the stack directly?
# does pop(r,n) copy to reg[n] or does reg[n] point to stack?
# is reg[n] a pointer to the top of the stack or actual storage?
# how do we deal with saving the inputs?

# all real storage is on the stack, there is no buffer
# when time increases some registers are pushed others overwritten
# so we need pushable registers with real storage
# reg(r,n) is just the top of stack(r,n)
# initially we have real space, i.e. stack is never empty
# when we push, stack increments its pointer
# net does not need to keep track of time, individual registers do

# can we have a single stack for all the registers?
# nobody keeps track of time, we hope push/pops balance
# where is the real storage?  still should be on stack and regs should just point.
# so we are pointing to and changing more than the top of the stack.
# where are the input registers pointing to?
# registers are all pointers!  none have real storage.
# input registers originally point to the last input.

net specifies which registers each op is going to use as input and output.
so before we do forw, we should point the input registers to the net input.
then we don't have to do anything special in forwinputs.

function forwinput(r::RNN, x, n)
    map(r.inputs[n]) do i               # return the input matrix or tuple of input matrices to produce r.output[n]
        i <= 0     ? x[1-i] :           # i <= 0 is used to indicate input x[1-i]
        i < n      ? r.output[i] :      # 0 < i < n are outputs from the current time slice
        r.time > 1 ? r.output[i] :	# i >= n are from the previous time slice if r.time > 1
        nothing                         # nothing represents zero matrix from r.time=0
    end
end

if we initialize registers to nothing, and copy the inputs, all will be fine and there will be no need for forwinput.

On the one hand we want reg[i]=nothing for uninitialized inputs
On the other hand we want reg[o]!=nothing for storage
Note that o==i is possible in which case we definitely want reg[o]==nothing initially
On the third hand we want to use the same storage for reg[o] from last minibatch
Keep an initial storage for each register?


###############
### I/O
###############

We need a good interface for models and data.  First data:

Earlier notes:
# Organization of the training set and training batches:
# D:dimensionality, I:instance, B:batch-instance, T:time
# In FFNN size(xtrain)=(D,I), size(xbatch)=(D,B) where D fastest
# In RNN size(xtrain)=(D,T,I) or xtrain=[(D,T1),(D,T2),...]
# i.e. we want each instance to be contiguous.
# In RNN size(xbatch)=(D,B,T) or xbatch=[(D,B1),(D,B2),...]
# i.e. we want each time-step to be contiguous.
# train->batch will need to do some shuffling

x[i][t][d1,d2,...] each instance and each token separate, default for rnn
x[d1,d2,...,i] this is the default minibatch for fnn
x[d...,i,t] possible minibatch for rnn?
x[d...,t,i] possible training set for rnn?

lowest level forw expects a single token: x[d...]
could also be a token-batch: x[d...,i]
this means if we have isbits(eltype(x)) we interpret it as a single token-batch.
this is consistent with fnn convention.
the time dimension is never embedded in a bits array.

if we are given x[t][d...,i] we interpret the first index as time.

if we are given x[i][t][d...,b] we interpret the first index as instance, second as time.

we can pull this off if fnn always sticks to bits arrays.

how do we test this:

(isbits(eltype(x))    ? first-type :
 isbits(eltype(x[1])) ? second-type :
 isbits(eltype(x[1][1])) ? third-type :
 error)

A = Union(Array{T<:Number}, Void)
B = Vector{A}
C = Vector{B}

We can distinguish Vector{Any} from the rest.
Specifying the bits arrays is more difficult there are too many types.
Low level forw can have a sanity check for eltype.
We can rename tforw -> forw(r,x::Vector{Any})

For forw(r,x[t][d...,i]) what is the expected output?
Loss: ok if training and y is provided but does not help predict.
also cannot compute loss after the fact unless we have access to outy array.
outy array: internal registers get overwritten, copying is expensive.
ask for external storage?
overwrite gold y and return loss?
return pointers to stack?

Low level forw(r,x) does not expect y, does not compute loss, returns internal storage.
For sequence forw we could be forgiven if we return internal storage since forw is
only supposed to be used internally and needs to be efficient.
This means we have to use the stack and construct a return Vector{Any} y array from internal storage.
- we don't know where in the stack stuff is stored.
- compiler may decide not to store the final out in stack.
- especially if train=false!

We copy external stuff in from input.  Makes sense if internal stuff got copied out.
Accept external storage with a keyword y parameter and overwrite if specified.
Predict works.
tforw works and passes the individual arrays to forw.
Do we return loss?  No, it is inefficient, not always necessary, and can be computed from the out y.
Do not return anything unless a y has been specified.


- train needs batching 
- predict needs batching
- sequence of batches need to be distinguished from sequence of instances
- currently train calls forw, back and update, i.e. one minibatch.
- updating loss and norm is going to be a problem.
- gclip needs to go to update.
- if x is a vector it is assumed to be a time sequence.

- sequence training data comes in x[i][t][d...]
- batch turns it into x[i][t][d...,b]
- adding feeds into train x[i], i.e. x[t][d...,b]
- train passes this to forw, back 
- forw takes it as a time sequence, runs init and passes x[d...,b] to forw1.

- mlp training data comes in as a block x[d...,i]
- old train batches this into x[i][d...,b]
- passes into forw/back x[d...,b]

- do we ever call train with x[i][d...]
- we might, expecting it to minibatch it like sequence.

- do we ever call train with x[t][d...]
- that is a single instance, probably not
- but x[t][d...,b] can be a batch?

- Array3 is always sequence data.
- Array1 is always block data. (if it is 1D it may be a single instance).
- Array2 is ambiguous.
- if x[i][d...] we need to batch into x[i/b][d...,b]
- if x[t][d...] it is a single instance. let's assume we don't call train with a single instance.
- if x[t][d...,i] we have already batched data, unlikely from the user.
- Disallow Array2?  Or interpret as x[i][d...]?  Or as x[i][d...,b]?

		block		sequence
instance	x[d...]		x[t][d...]
minibatch	x[d...,b]	x[t][d...,b]
dataset		x[d...,i]	x[i][t][d...]
dataset2	x[i][d...]


New language:

	* src/model/lstm2.jl: compiler args vs constructor args:
	lstm(10),x1,x2,y
	lstm(10),y,x1,x2
	y,lstm(10),x1,x2
	y=lstm(10)(x1,x2)
	y=lstm(x1,x2;h=10)
	Using the last convention with par the only exception.

# convention: subroutine constructors only take keyword args.
call these options to distinguish from parameters of the network.
# in machine code regular args are for the compiler, keyword args for constructor.
# par/rnd is the only exception, it has no regular input so dims can be regular args for readability.
par(n,0): convention: put 0 where the dimension is to be inferred from input.
# parameters, constants, and rand: all no-input ops
should we have par/rnd/con or should it all be options for par, i.e:
con is par with lr=0
rnd is par with rnd=f
how to specify an explicit value: val=array option?
if they were separate:
const: only sensible option is externally specified value: const([1]), const(1)
const will allow us to define things like 1-x, 0.5*x etc. without extra ops, using add, mul with smart broadcasting.
rnd: needs dimension, distribution and parameters, all except dimension can be keyword args.
it also needs test=false option.
subroutine constructors return quoted expressions.
these need to be compiled at some point maybe:
@net begin ... end: is a compiler: here is linear regression:

@net begin
    x = input()
    w = par(10,0)
    y = dot(w,x)
    b = par(0)
    z = add(b,y)
    r = qloss(z)	# use xloss/closs for cross-entropy loss?  should we make r optional here?
end

Do we need soft and softloss separately, can we combine them?  (soft is never used with any other loss and vice-versa)
We need to be careful and distinguish between ops that expect a gradient back vs ops that expect gold answers back.

Update is the part of the code that should be configurable.
Implement call-backs?

Where do we keep the matrices for adagrad, momentum etc?
forw/back don't care, these are for update only...
inside par, as usual?
is par going to write to registers out and dif?
we want to add a tmp register (replacing inc) but that is for transient values.

These are some big changes.  We need a plan, testing every step of the way.

### forw/back interface:
currently we have forw(l,x...;y=y)
we want to make y mandatory, get rid of on internal memory.
we can't have forw(l,x...,y)
option 1: forw(l,y,x...)
          back(l,dy,dx...)
if we are going to support a variable number of inputs, this seems to be the only way.
some dx may be nothing in which case they do not need computing.
back may need x/y or neither, these can be options?
option 2: forw(l,x,y) where x is a tuple, NO
option 3: forw(l,x1,x2,y) for two inputs, forw(l,x,y) for one, forw(l,y) for zero.
who initializes the weight matrices?  forw(par,y) does not know the size.
the user ops have to: forw(dot,x1,x2) during runtime.
but dot does not know which one if any is coming from par.
dot should also assume pre-initialized, and net should initialize before calling forw.

do we ever allow forw without y?  NO.
keyword args for back: x is a tuple?  when necessary.

option 2: forw(l,xy...) where the last of xy is picked up as the output.
          back(l,dy,dx...) 

does forw/back return anything?
what if they need to set their output to nothing?
not possible with this setup.  they can fill output with zeros.

what is the cudnn calling convention?
alpha,src,beta,dest
bias,src
src,dest
usually destination at the end.
that is also the convention in blas.
axpy(alpha,x,y)
opposite of julia
A_mul_B!(C,A,B)
we'll just go with option 1.


## Size Inference:

par starts life as an (n,0) array.  initforw should have enough
information to figure out what should replace 0.  we should not leave
size inference and par init to ops, take care of it in the net.  this
is similar to broadcasting inference.  inferrer needs to know the op
that will use the par and the other inputs to that op.  similar size
inference can help compiler reuse registers more effectively.

## rethinking initout:

we still want to overwrite input where we can but we also want to
reuse earlier arrays if possible.  initout is also responsible for
initializing the par arrays.  par arrays can come in as
(1) size with zeros
(2) complete size
(3) actual array
par types store their own arr, diff, or do we use ones stored in the
net?  if both, then the net ones should be pointers.  but probably no
need.  

ok, just use net.out and net.dif for par as well.  during comp just
store the dims, do not alloc.  accept actual array as well as rand
generator for init option (or have separate option).  during init we
alloc the actual array for par.  dif has to be sparse if input is
sparse.  actual dims depend on the operation for zero sizes.  eltype
is fixed throughout once given by the input.  if conflict with actual
par init array the input eltype wins.

infer eltype: easy, inputs win, single eltype throughout.
infer size: can do depending on op.
infer sparsity: also depends on input type and op.
infer sharing: can do after sizes are inferred.

do we do out/dif/tmp together?

can we have empty inputs and still init?
possibly, if pars are specified.

rnd does not come with an eltype, but par and con might.

sharing pattern is not the same for forw vs back?

initialization with arrays of par and con.

par(10,0) vs par(w) or par(init=w)
we have par(10,0; init=Gaussian(0,0.01))

call it p.out?  p.arr?  => p.arr or maybe p.init

par(x::Integer...; o...)=par(; dims=x, o...)
par(w::AbstractArray; o...)=par(; init=w, o...) vs par(; arr=w, o...)
all can set fields of par.
distinguish rand from init?  no.  init(par) can take care of both.

normal init should write to p.dims.

### size inference:
inputs are determined
par sizes are partly determined (with some zeros)
dot should be able to infer from x1=(n,0), x2=(a,b)
overwriters should copy their input sizes if fully determined
add/mul have to take into account broadcasting
who calls par.init?


dot,mul,add can do size inference with incomplete input sizes
dot => obvious
mul => scale vs elementwise
add => scale vs bias vs elementwise
let incomplete sizes be written to dims, then fixed.

in general do not change the ops when inferring size/type etc.
just create the registers with the correct size/type.

ISSUE:
who calls netinit?  is x enough?  cannot tell if y is sparse.
cannot complete infersparse for dif.
separate initforw and initback?


### calling convention
forw(op, y, x...): 
- should we keep y optional?  
- overwriting ops?
- calling convention for net?
back(op, dy, dx...)
- what if we want dx to point to dy?
- instead of copying?
loss(op, y, dy) or loss(op, dy, y)?

forw copying when no change?
back copying when no need?

change to forw(op, x..., y)
we can make y optional if we want to.


### sharing inference:
single tmp per size sufficient (necessary?)
forw_uses_tmp, back_uses_tmp?
each out0 has a read and write time.
first see if an input can be overwritten
next see if there is an array in garbage?
we know what each op reads and writes
we know what will be saved for back
overwrite the last read of possible
keep a stack of reads
every time something needs to be written go over the reads backwards
check whether each is available
size has to be write
not saved
not ever read again: until when?
not multi?
alloc if nothing available.
careful about read-before-write


### back sharing

1 --- a --- b --- N
considering overwriting b with a
b was read and possibly zeroed (if incr) 
b is read only once
b may be written on in:

bw: 1..a-1, a, a+1..b-1, b, b+1..N

a is going to be read once (now)
a may be written in:

aw: 1..a-1, a, a+1..b-1, b, b+1..N

thinking as a cycle, b is free between read at b and first write
a is busy between first write (going back from the read) and (only) read
it is ok to share if aw1...ar is contained in br...bw1 cyclically going back

example: regular mlp, bw1=b+1, br=b, so b is free everywhere

also additional sharing (may be used in other than b)

can a or b write themselves?  if it is its own input.  in which case
read comes first, the write.  
nowhere free or everywhere free?  nowhere free!

busy between first write and only read (going back)


"""
- do we need dif0?
- is the calling convention ok?
- tmp initialization
- no more params(r)
- do we ever want dx from the net? (for s2c type combos)
+ assume no dif0
- dy comes in, may be nothing, gets copied into dif[N]
- there is always a single output, a single dy
- other dif's are allocated but uninitialized
- computation always goes from dy[n] to dy[inputs[n]]
- could we ever have an uninitialized dy?
-- only if y[n] has not been an input for any y[n+k]
-- can be if y[n] has never been read or read by y[n-k]
-- if read by y[n-k] it will have dy set from prev iter except at t=T where dy=0
-- we can selectively zero those who need (unread and incr)
+ another reason for dif0 could be if we want dif to point to other things, i.e. add-back
- todo: back for scalar add, mul, check others
+ figure out dif sharing
- dif[n] is read once at iteration n, and dif[inputs[n]] are written to.
- dif[n] is free between the read and the first write.
- dif[n] is busy between the first write and the read.
+ todo: figure out sparse dw
- don't we need overwrites as well?  not all ops can overwrite! there is no point in sharing no-overwrite.
- can-overwrite: actf, add (to input 2), drop?, loss, 
- no-overwrite: conv, dot, mul, pool
- n/a: input, par
- mul: can overwrite input 2 going forward in element-wise or scaled
-- going back we need both x[1] and x[2] so really no-forw-overwrite
-- back could possibly overwrite dy with dx2 but not worth the trouble
- add: can overwrite 1 or 2 going forw element-wise
-- only input 2 if bias or scalar add but the size will tell us
-- back does not read anything so no problem either way
- is overwrites == !back_reads_x?  yes, except for par and input for
which the question is invalid.  this may not be always true, sizes
have to match etc etc.
- is forw.overwrites == back.overwrites?
-- ow=true: actf:yes add=yes drop=? loss=yes
-- ow=fals: conv, dot, pool change size so no question, mul cannot.
-- most importantly we forgot: overwriting is good if it avoids
copying.
"""

## Interface issues:

op interface:
forw(op, x..., y)
back(op, dy, dx...; x, y)

net interface:
forw(net, x...; yout, ygold)
back(net, dy; dx)
