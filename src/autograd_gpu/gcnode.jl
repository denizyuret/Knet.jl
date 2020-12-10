using CUDA: CuArray, unsafe_free!
using Knet.CuArrays: cuarrays
using Knet.KnetArrays: cuallocator
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

# The gcnode_queue maps CuArrays to the first index on tape they have a reference in. We use
# Base.Order.Reverse because we want to free the pointers with highest indices first.  Using
# ObjectId to allow fast hashing, hashing CuArrays directly is slow like Arrays.
const gcnode_queue = PriorityQueue{UInt,Int}(Base.Order.Reverse)
    
# To call unsafe_free gcnode_dict helps us find which CuArray belongs to an objectid.  Using
# WeakRef on CuArray values to allow garbage collection. Note that as a result sometimes
# WeakRef.value will metamorphose from CuArray to nothing if gc gets to it earlier.
const gcnode_dict = Dict{UInt,WeakRef}()

# We use node indices on the tape to use as values in the priority queue. Using
# ObjectId(::Node) for keys to allow gc after gcnode is done.
const gcnode_index = Dict{UInt,Int}()


# During the backward step parents of a node (who have lower indices) may have their
# outgrads modified, thus new references to CuArrays that we have already indexed may
# appear. We want to keep the smallest index for each CuArray.
function gcnode_setindex!(c::CuArray,v::Int)
    cid = objectid(c)
    get!(gcnode_dict, cid) do; WeakRef(c); end
    if v < get(gcnode_queue,cid,typemax(Int))
        gcnode_queue[cid] = v
    end
end

function gcnode_init(tape::Tape)
    empty!(gcnode_index)
    empty!(gcnode_queue)
    empty!(gcnode_dict)
    @inbounds for (i,n) in enumerate(tape.list)
        gcnode_index[objectid(n)] = i
        if n.Value isa Result
            for c in cuarrays(n.Value.value); gcnode_setindex!(c,0); end # pointers with index 0 will never get gc'ed
            for c in cuarrays(n.outgrad);     gcnode_setindex!(c,i); end # this only sets gcnode_queue[c] if it was not seen
        else # n.Value isa Param
            for c in cuarrays(n.Value.value); gcnode_setindex!(c,0); end
            for c in cuarrays(n.outgrad);     gcnode_setindex!(c,0); end
        end
    end
    # Mark this tape so we know when gcnode is called with a new tape and gcnode_init is needed
    tape.dict[gcnode_null] = gcnode_null_node
end


function gcnode(n::Node, tape::Tape)
    cuallocator[] || return knetgcnode(n,tape)
    if !haskey(tape.dict, gcnode_null)
        gcnode_init(tape)
    end
    ni = gcnode_index[objectid(n)]
    if n.Value isa Result
        for c in cuarrays(n.outgrad); gcnode_setindex!(c, ni); end
    end
    @inbounds for i in 1:length(n.parents);
        isassigned(n.parents, i) || continue
        parent = n.parents[i]
        if parent.Value isa Result
            pi = gcnode_index[objectid(parent)]
            for c in cuarrays(parent.outgrad); gcnode_setindex!(c, pi); end
        else
            for c in cuarrays(parent.outgrad); gcnode_setindex!(c,0); end # protect Params
        end
    end
    while !isempty(gcnode_queue) && peek(gcnode_queue)[2] >= ni  ## 5.62μs
        (cid,v) = dequeue_pair!(gcnode_queue)
        @assert v == ni
        c = gcnode_dict[cid].value
        if c isa CuArray ## c turns into nothing if gc'ed
            unsafe_free!(c)
        end
    end
    if n.Value isa Result
        n.Value, n.outgrad = gcnode_null, nothing
    end
end


const gcnode_null = Result{Nothing}(nothing,nothing,nothing,nothing)
const gcnode_null_node = Node(gcnode_null)
