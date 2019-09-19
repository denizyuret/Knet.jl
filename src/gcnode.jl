# During the back pass we want to make pointers available as soon as we can to save memory
# without waiting for gc. This is risky as we have to make sure the pointers are not going
# to be used again.  We initialize a priority queue in knetgcinit() where KnetPtrs are
# mapped to the last index of the tape they are used.  If a KnetPtr is part of a Param we do
# not want to free it, so we map those to typemax(Int).  Every time knetgcnode() is called
# after a back step, we update the queue with the pointers from the new outgrads of the node
# and its parents. The values in the queue are only increased, never decreased. Finally we
# free the pointers whose queue value is the current tape position. Note that tape indices
# go backward in time from loss to parameters.

using DataStructures: PriorityQueue, dequeue!, dequeue_pair!, DataStructures
using AutoGrad: Node, Tape, Result

_tape = nothing
_index = IdDict{Node,Int}()
_queue = PriorityQueue{KnetPtr,Int}()
incval!(q::PriorityQueue{KnetPtr,Int},k::KnetPtr,v::Int) = if get(q,k,0) < v; q[k]=v; end  ## 0.190μs

function knetgcinit(tape)  ## 2.35ms
    global _tape, _index, _queue
    _tape = WeakRef(tape)
    empty!(_index)
    empty!(_queue)
    tape isa Tape || return
    @inbounds for (i,n) in enumerate(tape.list)
        _index[n] = i
        index = (n.Value isa Result ? i : typemax(Int))
        for k in knetptrs(n.Value.value); incval!(_queue,k,index); end ## knetptrs: 0.283μs
        for k in knetptrs(n.outgrad); incval!(_queue,k,index); end     ## incval: 0.293μs
    end
end

function knetgcnode(n::Node, tape=nothing)  ## 16.3μs
    tape != _tape && knetgcinit(tape) ## 2μs amortized
    tape isa Tape || return
    ni = _index[n]
    if n.Value isa Result && n.outgrad isa KnetArray
        incval!(_queue, n.outgrad.ptr, ni)
    end
    @inbounds for i in 1:length(n.parents);  ## 2.43μs
        isassigned(n.parents, i) || continue
        parent = n.parents[i]
        pi = (parent.Value isa Result ? _index[parent] : typemax(Int)) # protect Params
        for ptr in knetptrs(parent.outgrad)      ## 0.680μs
            incval!(_queue, ptr, pi)             ## 0.486μs
        end
    end
    while !isempty(_queue) && DataStructures.peek(_queue)[2] <= ni  ## 5.62μs
        (k,v) = dequeue_pair!(_queue)  ## 0.787μs
        if v != ni; @warn("k=$((k.ptr,k.len)) v=$v ni=$ni", maxlog=1); end  ## 0.160μs
        #DBG verifypointer(tape, ni, k) 
        freeKnetPtr(k)  ## 4.06μs
    end
    if n.Value isa Result
        n.outgrad = n.Value.value = nothing
    end
end


# Recursively search for KnetPtrs based on deepcopy_internal

knetptrs(x, c=KnetPtr[], d=IdDict{Any,Bool}()) = (_knetptrs(x,c,d); c)

_knetptrs(x::Tuple, c::Vector{KnetPtr}, d::IdDict{Any,Bool}) =
    for xi in x; _knetptrs(xi, c, d); end

_knetptrs(x::Union{Module,String,Symbol,Core.MethodInstance,Method,GlobalRef,DataType,Union,UnionAll,Task,Regex},
          c::Vector{KnetPtr}, d::IdDict{Any,Bool}) = return

_knetptrs(x::Core.SimpleVector, c::Vector{KnetPtr}, d::IdDict{Any,Bool}) =
    if !haskey(d,x); d[x] = true; for xi in x; _knetptrs(xi, c, d); end; end

_knetptrs(x::Array, c::Vector{KnetPtr}, d::IdDict{Any,Bool}) =
    if !haskey(d,x); d[x] = true; _knetptrs_array_t(x, eltype(x), c, d); end

_knetptrs(x::Union{Dict,IdDict}, c::Vector{KnetPtr}, d::IdDict{Any,Bool}) =
    if !haskey(d,x); d[x] = true; for (k,v) in x; _knetptrs(k, c, d); _knetptrs(v, c, d); end; end

function _knetptrs_array_t(@nospecialize(x), T, c::Vector{KnetPtr}, d::IdDict{Any,Bool})
    if isbitstype(T)
        return
    end
    for i = 1:(length(x)::Int)
        if ccall(:jl_array_isassigned, Cint, (Any, Csize_t), x, i-1) != 0
            xi = ccall(:jl_arrayref, Any, (Any, Csize_t), x, i-1)
            if !isbits(xi)
                _knetptrs(xi, c, d)
            end
        end
    end
end

function _knetptrs(@nospecialize(x), c::Vector{KnetPtr}, d::IdDict{Any,Bool})
    T = typeof(x)::DataType
    nf = nfields(x)
    (isbitstype(T) || nf == 0) && return
    if haskey(d, x)
        return
    end
    if T.mutable
        d[x] = true
    end
    if T === KnetPtr
        push!(c, x)
    end
    for i in 1:nf
        if isdefined(x,i)
            _knetptrs(getfield(x,i), c, d)
        end
    end
end


## Old implementation: gets about half the pointers at about twice the speed (but the speed advantage is negligible)

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


## Debugging utilities

gcpointers = [ 0., 0., 0., 0. ]     #DBG

function verifypointer(tape::Tape, ni::Int, k::KnetPtr) # for debugging
    @show ni
    gcpointers[1] += k.len
    findk = false
    for n in tape.list
        if k in knetptrs(n.outgrad) || k in knetptrs(n.Value.value)
            findk = true
            @assert n.Value isa Result && n.Value.value isa KnetArray && n.outgrad isa KnetArray  (global _n=n; global _k=k; global _t=tape; "pointer $(k.ptr) found in $n")
        end
    end
    @assert findk
    for i in 1:length(tape.list)
        p = tape.list[i]
        if p.Value isa Param || i > ni
            @assert !isa(p.Value.value,KnetArray) || (p.Value.value.ptr != k && p.Value.value.ptr.ptr != C_NULL && p.Value.value.ptr.ptr != k.ptr)  (global _i=i; global _p=p; global _n=tape.list[ni]; global _ni=ni; "null value")
            @assert !isa(p.outgrad,KnetArray) || (p.outgrad.ptr != k && p.outgrad.ptr.ptr != C_NULL && p.outgrad.ptr.ptr != k.ptr)  (global _i=i; global _p=p; global _n=tape.list[ni]; global _ni=ni; "null outgrad")
        end
    end
end
