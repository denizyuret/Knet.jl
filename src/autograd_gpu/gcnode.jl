using CUDA: CuArray, unsafe_free!
using Knet.CuArrays: cuarrays
using AutoGrad: Result, Node, Tape

# During the back pass we want to make pointers available as soon as we can to save memory
# without waiting for gc. This is risky as we have to make sure the pointers are not going to
# be used again.  We initialize a priority queue in gcnode_init() where CuArrays are mapped to
# the smallest index of the tape they are used.  If a CuArray is in a Param.value,
# Result.value, or Node{Param}.outgrad we do not want to free it, so we map those to 0. Only
# Node{Result}.outgrads can be safely freed, Result values are not safe because the user can
# assign them to global variables (as is done in RNNs).  Every time gcnode() is called after a
# back step, we update the queue with the pointers from the new outgrads of the node and its
# parents. The values in the queue are only decreased, never increased. Finally we free the
# pointers whose queue value is the current tape position. Note that tape indices go forward
# in time from parameters to loss but are processed backward in time from loss to parameters.

# The gcnode_queue maps CuArrays to the first index on tape they have a reference to. We use
# Base.Order.Reverse because we want to free the pointers with highest indices first.
# Using WeakRef to allow garbage collection.
const gcnode_queue = PriorityQueue{WeakRef,Int}(Base.Order.Reverse)
    
# During the backward step parents of a node (who have lower indices) may have their
# outgrads modified, thus new CuArray references may appear. We want to keep the smallest
# index for each CuArray. 
function gcnode_minidx!(q::PriorityQueue{WeakRef,Int,typeof(Base.Order.Reverse)},k::CuArray,v::Int)
    if v < get(q,k,typemax(Int)); q[WeakRef(k)]=v; end  ## 0.190μs
end

const gcnode_index = WeakKeyDict{Node,Int}()
gcnode_tape = WeakRef(nothing)

function gcnode_init(tape::Tape)  ## 2.35ms
    global gcnode_tape, gcnode_index, gcnode_queue
    gcnode_tape = WeakRef(tape)
    empty!(gcnode_index)
    empty!(gcnode_queue)
    tape isa Tape || return
    @inbounds for (i,n) in enumerate(tape.list)
        gcnode_index[n] = i
        if n.Value isa Result
            for k in cuarrays(n.Value.value); gcnode_queue[WeakRef(k)] = 0; end # pointers with index 0 will never get gc'ed
            for k in cuarrays(n.outgrad);  get!(gcnode_queue,WeakRef(k),i); end # this only sets gcnode_queue[k] if it does not have a value
        else # n.Value isa Param
            for k in cuarrays(n.Value.value); gcnode_queue[WeakRef(k)] = 0; end
            for k in cuarrays(n.outgrad);     gcnode_queue[WeakRef(k)] = 0; end
        end
    end
end

function gcnode(n::Node, tape::Tape)  ## 16.3μs
    global gcnode_tape, gcnode_index, gcnode_queue
    tape !== gcnode_tape.value && gcnode_init(tape) ## 2μs amortized
    tape isa Tape || return
    ni = gcnode_index[n]
    if n.Value isa Result # && n.outgrad isa KnetArray
        for ptr in cuarrays(n.outgrad); gcnode_minidx!(gcnode_queue, ptr, ni); end
    end
    @inbounds for i in 1:length(n.parents);  ## 2.43μs
        isassigned(n.parents, i) || continue
        parent = n.parents[i]
        if parent.Value isa Result
            pi = gcnode_index[parent]
            for ptr in cuarrays(parent.outgrad); gcnode_minidx!(gcnode_queue, ptr, pi); end
        else
            for ptr in cuarrays(parent.outgrad); gcnode_queue[WeakRef(ptr)] = 0; end # protect Params
        end
    end
    while !isempty(gcnode_queue) && peek(gcnode_queue)[2] >= ni  ## 5.62μs
        (k,v) = dequeue_pair!(gcnode_queue)  ## 0.787μs
        k = k.value
        if v != ni; @warn("k=$((k.ptr,k.len)) v=$v ni=$ni", maxlog=1); end  ## 0.160μs
        #DBG verifypointer(tape, ni, k) 
        unsafe_free!(k)  ## 4.06μs
    end
    if n.Value isa Result
        n.Value, n.outgrad = gcnode_null, nothing
    end
end

const gcnode_null = Result{Nothing}(nothing,nothing,nothing,nothing)
