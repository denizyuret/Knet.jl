# Eliminate Combinatorics dependency. These are based on Julia4 source.

import Base: eltype, length, iterate 

struct Combinas{T}
    a::T
    t::Int
end

eltype(::Type{Combinas{T}}) where {T} = Vector{eltype(T)}

length(c::Combinas) = binomial(length(c.a),c.t)

function combinas(a, t::Integer)
    if t < 0
        # generate 0 combinations for negative argument
        t = length(a)+1
    end
    Combinas(a, t)
end

function iterate(c::Combinas, s=[1:c.t;])
    if !isempty(s) && s[1] > length(c.a)-c.t+1; return nothing; end
    comb = [c.a[si] for si in s]
    if c.t == 0
        # special case to generate 1 result for t==0
        return (comb,[length(c.a)+2])
    end
    s = copy(s)
    for i = length(s):-1:1
        s[i] += 1
        if s[i] > (length(c.a) - (length(s)-i))
            continue
        end
        for j = i+1:endof(s)
            s[j] = s[j-1]+1
        end
        break
    end
    (comb,s)
end

struct Permutas{T}
    a::T
end

eltype(::Type{Permutas{T}}) where {T} = Vector{eltype(T)}

length(p::Permutas) = factorial(length(p.a))

permutas(a) = Permutas(a)

function iterate(p::Permutas, s=[1:length(p.a);])
    if !isempty(s) && s[1] > length(p.a); return nothing; end
    perm = [p.a[si] for si in s]
    if length(p.a) == 0
        # special case to generate 1 result for len==0
        return (perm,[1])
    end
    s = copy(s)
    k = length(s)-1
    while k > 0 && s[k] > s[k+1];  k -= 1;  end
    if k == 0
        s[1] = length(s)+1   # done
    else
        l = length(s)
        while s[k] >= s[l];  l -= 1;  end
        s[k],s[l] = s[l],s[k]
        reverse!(s,k+1)
    end
    (perm,s)
end
