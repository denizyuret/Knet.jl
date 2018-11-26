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

using AutoGrad: Node

if isdefined(AutoGrad,:gcnode)  # TODO: remove after 1.1.1
    function AutoGrad.gcnode(n::Node)
        if maybefree(n.outgrad, n); n.outgrad = nothing; end
        if maybefree(n.Value.value, n); n.Value.value = nothing; end
        # n.outgrad=n.Value.value=nothing # this prevents later shared pointers from being discovered
    end
end

maybefree(x,n)=false

function maybefree(x::KnetArray, n::Node)
    @inbounds for i in 1:length(n.parents)
        if isassigned(n.parents, i) && (maysharepointer(x, n.parents[i].outgrad) || maysharepointer(x, n.parents[i].Value.value)) # need to check both outgrad and value
            return false
        end
    end
    @inbounds for r in n.children
        if maysharepointer(x, r.outgrad) || maysharepointer(x, r.Value.value)
            return false
        end
    end
    #DBG
    # cp = countpointer(x)
    # if cp != 1
    #     global badx = x
    #     error("Missed shared pointer $cp")
    # end
    #DBG
    # CuArrays.unsafe_free!(x.ptr)
    return true
end

# This returns false only if we are sure there is no shared pointer. It is conservative, may return true when it shouldn't.
# Numbers, Nothing, unshared KnetArray with different pointer (98%) is safe.
maysharepointer(x::KnetArray, p)=!(isbits(p) || (isa(p, KnetArray) && !isdefined(p,:parent) && pointer(p) != pointer(x)))

function countpointer(x::KnetArray)
    cnt = 0
    for n in AutoGrad.lasttape
        if isa(n.outgrad,KnetArray) && pointer(n.outgrad) == pointer(x); cnt+=1; end
        if isa(n.Value.value,KnetArray) && pointer(n.Value.value) == pointer(x); cnt+=1; end
    end
    return cnt
end
