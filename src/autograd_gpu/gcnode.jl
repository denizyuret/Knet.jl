using CUDA: CuArray, unsafe_free!
using Knet.CuArrays: cuarrays  # Profiling shows most of the time is spent in cuarrays
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

# There is no explicit call to gcnode_init by the user. gcnode just resets everything if the
# input Tape is different from gcnode_tape:
const gcnode_tape = WeakRef(nothing)
const gcnode_tape_id = Ref(UInt(0))

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
    gcnode_tape.value = tape
    gcnode_tape_id[] = objectid(tape)
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
end

function gcnode(n::Node, tape::Tape)
    cuallocator[] || return knetgcnode(n,tape)
    if tape !== gcnode_tape.value
        gcnode_init(tape)
    elseif objectid(tape) !== gcnode_tape_id[]
        println("Tape inconsistency: objectid(tape)=$(objectid(tape)) gcnode_tape_id[]=$(gcnode_tape_id[]) objectid(gcnode_tape.value) = $(objectid(gcnode_tape.value))")
    end
    @inbounds for i in 1:length(n.parents);
        isassigned(n.parents, i) || continue
        parent = n.parents[i]
        if parent.Value isa Result
            pi = get(gcnode_index, objectid(parent), 0)
            if pi > 0
                for c in cuarrays(parent.outgrad); gcnode_setindex!(c, pi); end
            else
                gcnode_debug_parent(n, tape, i)
            end
        else
            for c in cuarrays(parent.outgrad); gcnode_setindex!(c,0); end # protect Params
        end
    end
    ni = get(gcnode_index, objectid(n), 0)
    if ni == 0
        gcnode_debug_node(n, tape)
        return
    end
    if n.Value isa Result
        for c in cuarrays(n.outgrad); gcnode_setindex!(c, ni); end
    end
    while !isempty(gcnode_queue) && peek(gcnode_queue)[2] >= ni  ## 5.62μs
        (cid,v) = dequeue_pair!(gcnode_queue)
        c = gcnode_dict[cid].value
        if v == ni && c isa CuArray  ## c turns into nothing if gc'ed
            unsafe_free!(c)
        end
    end
    if n.Value isa Result
        n.Value, n.outgrad = gcnode_null, nothing
    end
end

const gcnode_null = Result{Nothing}(nothing,nothing,nothing,nothing)

function gcnode_debug_node(n, tape)
    println("WARNING: Cannot find node $(objectid(n)) in gcnode_index")
    println("objectid(tape)=$(objectid(tape)) gcnode_tape_id[]=$(gcnode_tape_id[]) objectid(gcnode_tape.value) = $(objectid(gcnode_tape.value))")
    ni = findfirst(isequal(n), tape.list)
    if ni === nothing
        println("The node does not appear on tape.list")
    else
        println("The node appears on index $ni on tape.list")
    end
    nj = findfirst(isequal(n), tape.dict)
    if nj === nothing
        println("The node does not appear on tape.dict")
    else
        println("The node appears on tape.dict")
    end
    @show sort(collect(keys(gcnode_index))) == sort(objectid.(tape.list))
    @show sort(collect(keys(gcnode_index))) == sort(objectid.(collect(values(tape.dict))))
    @show sort(objectid.(tape.list)) == sort(objectid.(collect(values(tape.dict))))
    println("gcnode_index has $(length(gcnode_index)) objectid(node) keys:")
    println(collect(keys(gcnode_index)))
    println("tape.list has $(length(tape.list)) nodes:")
    println(objectid.(tape.list))
    println("tape.dict has $(length(tape.dict)) Tracked-Node pairs:")
    println(objectid.(collect(values(tape.dict))))
end

function gcnode_debug_parent(n, tape, i)
    parent = n.parents[i]
    println("WARNING: Cannot find parent $i of $(objectid(n)) with id $(objectid(parent)) in gcnode_index")
    println("objectid(tape)=$(objectid(tape)) gcnode_tape_id[]=$(gcnode_tape_id[]) objectid(gcnode_tape.value) = $(objectid(gcnode_tape.value))")
    ni = findfirst(isequal(parent), tape.list)
    if ni === nothing
        println("The parent does not appear on tape.list")
    else
        println("The parent appears on index $ni on tape.list")
    end
    nj = findfirst(isequal(parent), tape.dict)
    if nj === nothing
        println("The parent does not appear on tape.dict")
    else
        println("The parent appears on tape.dict")
    end
    @show sort(collect(keys(gcnode_index))) == sort(objectid.(tape.list))
    @show sort(collect(keys(gcnode_index))) == sort(objectid.(collect(values(tape.dict))))
    @show sort(objectid.(tape.list)) == sort(objectid.(collect(values(tape.dict))))
    println("gcnode_index has $(length(gcnode_index)) objectid(node) keys:")
    println(collect(keys(gcnode_index)))
    println("tape.list has $(length(tape.list)) nodes:")
    println(objectid.(tape.list))
    println("tape.dict has $(length(tape.dict)) Tracked-Node pairs:")
    println(objectid.(collect(values(tape.dict))))
end
