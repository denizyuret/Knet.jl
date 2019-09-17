# During the back pass we want to make pointers available as soon as we can to save memory
# without waiting for gc. This is risky as we have to make sure the pointers are not going
# to be used again.  We want to make sure there are no shared pointers with parents or
# children in the computational graph. Results only get created at core.jl:100 in forw().
# We have f, args, kwargs recorded so we have all the parents. The children are later
# Results and outgrads of parents. We have no direct access to children!  Outgrads are only
# created in core.jl:74 in back(). Both parents and children are accessible from the node.

# TODO: remove shared pointers after last use
# TODO: improve maysharepointer
# TODO: possibly do a full tape scan in the beginning
# TODO: play with other blocksize functions

# Currently we free only 1/3 of possible space.
# We only want to track/free KnetArrays
# Say we have a function to recursively search for KnetArrays.
# We can have a last-used hash KnetArray->node.

using DataStructures: PriorityQueue, dequeue!, dequeue_pair!, DataStructures
using AutoGrad: Node, Tape, Result

_tape = nothing
_nodes = IdDict{Node,Int}()
_kptrs = PriorityQueue{KnetPtr,Int}()
_param = IdDict{KnetPtr,Bool}()

function knetgcinit(tape)
    global _tape, _nodes, _kptrs
    _tape = WeakRef(tape)
    empty!(_nodes)
    empty!(_kptrs)
    if tape isa Tape
        @inbounds for i in 1:length(tape.list)
            node = tape.list[i]
            _nodes[node] = i
            index = (node.Value isa Result ? i : typemax(Int)) # protect Params
            # In case of shared pointers, larger values of i will override smaller ones
            for p in knetptrs(node.Value.value)
                _kptrs[p] = index
            end
            for p in knetptrs(node.outgrad)
                _kptrs[p] = index
            end
        end
    end
end

function knetgcnode(n::Node, tape=nothing)  # knetgcnode: 191ms, 16.5μs/call
    n.Value isa Result || return
    tape != _tape && knetgcinit(tape) # knetgcinit: 33.0ms
    @inbounds for i in 1:length(n.parents)  # forloop: 60.8ms
        isassigned(n.parents, i) || continue
        p = n.parents[i]
        index = (p.Value isa Result ? _nodes[p] : typemax(Int)) # protect Params
        for ptr in knetptrs(p.outgrad)      # knetptrs: 36.7ms
            if get(_kptrs,ptr,0) < index    # get_kptrs: 3.16ms
                _kptrs[ptr] = index         # set_kptrs: 7.39ms
            end
        end
    end
    index = _nodes[n]
    while !isempty(_kptrs) && DataStructures.peek(_kptrs)[2] <= index # while: 59.3ms
        (k,v) = dequeue_pair!(_kptrs)  # dequeue: 10.0ms
        if v != index; @warn("k=$((k.ptr,k.len)) v=$v index=$index", maxlog=1); end
        freeKnetPtr(k)                 # freeKnetPtr: 36.3ms
    end
    n.outgrad = n.Value.value = nothing
end

# Recursively search for KnetPtrs based on deepcopy_internal

knetptrs(f) = (ps=KnetPtr[]; _knetptrs(f,ps,IdDict{Any,Bool}()); ps)

_knetptrs(p::KnetPtr, ps::Vector{KnetPtr}, d::IdDict{Any,Bool}) = if !haskey(d,p); d[p]=true; push!(ps,p); end

_knetptrs(x::Union{Symbol,Core.MethodInstance,Method,GlobalRef,DataType,Union,UnionAll,Task,Regex},
          ps::Vector{KnetPtr}, stackdict::IdDict{Any,Bool}) = return
_knetptrs(x::Tuple, ps::Vector{KnetPtr}, stackdict::IdDict{Any,Bool}) =
    for p in x; _knetptrs(p, ps, stackdict); end

_knetptrs(x::Module, ps::Vector{KnetPtr}, stackdict::IdDict{Any,Bool}) = return

_knetptrs(x::String, ps::Vector{KnetPtr}, stackdict::IdDict{Any,Bool}) = return

function _knetptrs(x::Core.SimpleVector, ps::Vector{KnetPtr}, stackdict::IdDict{Any,Bool})
    if haskey(stackdict, x)
        return
    end
    stackdict[x] = true
    for p in x; _knetptrs(p, ps, stackdict); end
end

function _knetptrs(@nospecialize(x), ps::Vector{KnetPtr}, stackdict::IdDict{Any,Bool})
    T = typeof(x)::DataType
    nf = nfields(x)
    (isbitstype(T) || nf == 0) && return
    if haskey(stackdict, x)
        return
    end
    if T.mutable
        stackdict[x] = true
    end
    for i in 1:nf
        if isdefined(x,i)
            _knetptrs(getfield(x,i), ps, stackdict)
        end
    end
end

function _knetptrs(x::Array, ps::Vector{KnetPtr}, stackdict::IdDict{Any,Bool})
    if haskey(stackdict, x)
        return
    end
    _knetptrs_array_t(x, eltype(x), ps, stackdict)
end

function _knetptrs_array_t(@nospecialize(x), T, ps::Vector{KnetPtr}, stackdict::IdDict{Any,Bool})
    stackdict[x] = true
    if isbitstype(T)
        return
    end
    for i = 1:(length(x)::Int)
        if ccall(:jl_array_isassigned, Cint, (Any, Csize_t), x, i-1) != 0
            xi = ccall(:jl_arrayref, Any, (Any, Csize_t), x, i-1)
            if !isbits(xi)
                xi = _knetptrs(xi, ps, stackdict)
            end
        end
    end
end

function _knetptrs(x::Union{Dict,IdDict}, ps::Vector{KnetPtr}, stackdict::IdDict{Any,Bool})
    if haskey(stackdict, x)
        return
    end
    stackdict[x] = true
    for (k, v) in x
        _knetptrs(k, ps, stackdict)
        _knetptrs(v, ps, stackdict)
    end
end


# Old implementation: gets about half the pointers at about twice the speed (but the speed advantage is negligible)

function knetgcnode_old(n::Node, tape=nothing)
    if maybefree(n.outgrad, n, tape); n.outgrad = nothing; end
    if maybefree(n.Value.value, n, tape); n.Value.value = nothing; end
    # n.outgrad=n.Value.value=nothing # this prevents later shared pointers from being discovered
end

maybefree(x,n,t)=false

function maybefree(x::KnetArray, n::Node, tape::Tape)
    # cp = countpointer(x, tape) #DBG
    @inbounds for i in 1:length(n.parents)
        isassigned(n.parents, i) || continue
        p = n.parents[i]
        if maysharepointer(x, p.outgrad) || maysharepointer(x, p.Value.value) # need to check both outgrad and value
            # gcpointers[cp > 1 ? 2 : 3] += 1 #DBG
            # gcpointers[2] += x.ptr.len #DBG
            return false
        end
    end
    @inbounds for r in n.children
        if maysharepointer(x, r.outgrad) || maysharepointer(x, r.Value.value)
            # gcpointers[cp > 1 ? 2 : 3] += 1 #DBG
            # gcpointers[2] += x.ptr.len #DBG
            return false
        end
    end
    @dbg (push!(arraysizes,0); push!(blocksizes,0))
    # gcpointers[cp > 1 ? 4 : 1] += 1 #DBG
    # gcpointers[1] += x.ptr.len #DBG
    freeKnetPtr(x.ptr)
    return true
end

# This returns false only if we are sure there is no shared pointer. It is conservative, may return true when it shouldn't.
# Numbers, Nothing, unshared KnetArray with different pointer (98%) is safe.
function maysharepointer(x::KnetArray, y)
    # !(isbits(y) || (isa(y, KnetArray) && !isdefined(y.ptr,:parent) && pointer(y) != pointer(x)))
    if isbits(y)
        return false            # primitive number types do not share
    elseif !isa(y, KnetArray)
        return true             # data structures may share
    elseif isa(y.ptr.parent, KnetPtr) || isa(x.ptr.parent, KnetPtr)
        return true             # if one is a shared array they may share pointers
    else 
        return (pointer(x) == pointer(y))
    end
end

function countpointer(x::KnetArray, tape::Tape)
    cnt = 0
    for n in tape.list
        if isa(n.outgrad,KnetArray) && pointer(n.outgrad) == pointer(x); cnt+=1; end
        if isa(n.Value.value,KnetArray) && pointer(n.Value.value) == pointer(x); cnt+=1; end
    end
    return cnt
end

gcpointers = [ 0., 0., 0., 0. ]     #DBG
