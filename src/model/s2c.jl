# S2C: sequence to class model

import KUnet: params, loss, forw, back

immutable S2C <: Model; net1; net2; params;
    S2C(a,b)=new(a,b,vcat(params(a),params(b)))
end

params(r::S2C)=r.params

loss(r::S2C,y; a...)=loss(r.net2,y; a...)

function forw(r::S2C, x::Vector; y=nothing, a...)
    n = nops(r.net1)            
    forw(r.net1, x; a...)
    forw(r.net2, r.net1.out[n]; y=y, a...)
    # TODO: implement lastout or direct write from net1 out to net2 input
    # You can do the latter if net2 is initialized first or just overwrite its buf0
    # and check for === before copy
end

function back(r::S2C, y; a...)
    back(r.net2, y; a...)
    initback(r.net1; seq=true, a...)
    back(r.net1, r.net2.dif[nops(r.net2)+1]; seq=true, a...)
    while r.net1.sp > 0; back(r.net1, nothing; seq=true, a...); end
end

