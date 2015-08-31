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


### Can we have a recursive Net type?

; need to consolidate the types RNN, Net, and Layer.
; do we append or list when merging?

### Can we have reversible RNNs?

;;;; can the x's be recalculated using a reverse forw function?
;;;; no, add2 and mul2 lose information 
; although we could have two outputs with two inputs and make them reversible
; however this will not prevent having to remember something for each time step of forw?
;;;; mmul loses information unless w is square.
;;;; bias is reversible.
;;;; tanh/sigm reversible, relu loses information.
;;;; i think it is worth thinking about reversible rnn's.

