# S2C: sequence to class model

import Knet: params, forw, back
using Knet: initback

immutable S2C <: Model; net1; net2; params;
    S2C(a,b)=new(a,b,vcat(params(a),params(b)))
end

params(r::S2C)=r.params

function forw(r::S2C, x::Vector, yout=nothing; ygold=nothing, a...)
    forw(r.net1, x; a...)
    forw(r.net2, r.net1.out[end], yout; ygold=ygold, a...)
    # TODO: implement lastout or direct write from net1 out to net2 input
    # You can do the latter if net2 is initialized first or just overwrite its buf0
    # and check for === before copy
end

function back(r::S2C, y; a...)
    dy = similar(r.net1.out[end])
    back(r.net2, y, dy; a...)
    initback(r.net1, dy; seq=true, a...)
    back(r.net1, dy; seq=true, a...)
    while r.net1.sp > 0; back(r.net1, nothing; seq=true, a...); end
end

