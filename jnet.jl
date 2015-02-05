module Jnet

const libjnet = "./libjnet.so"
typealias Layer Ptr{Void}
typealias Dtype Float32
typealias Net Array{Layer,1}
typealias Mat AbstractMatrix{Dtype}
typealias Vec AbstractVector{Dtype}
typealias Cnet Ptr{Layer}
typealias Cmat Ptr{Dtype}

lfree(l::Layer)=ccall((:lfree,libjnet), Void, (Layer,), l)
lsize(l::Layer,i::Int)=Int(ccall((:lsize,libjnet), Cint, (Layer,Cint), l, i))
lsize(l::Layer)=(lsize(l,1),lsize(l,2))
relu(w::Mat,b::Vec)=ccall((:relu,libjnet), Layer, (Cint,Cint,Cmat,Cmat), size(w,1), size(w,2), w, b)
soft(w::Mat,b::Vec)=ccall((:soft,libjnet), Layer, (Cint,Cint,Cmat,Cmat), size(w,1), size(w,2), w, b)
forward(net::Net, x::Mat, y::Mat, batch::Int)=ccall((:forward,libjnet), Void, (Cnet, Cmat, Cmat, Cint, Cint, Cint), net, x, y, length(net), size(x,2), batch)
forwback(net::Net, x::Mat, y::Mat, batch::Int=100)=ccall((:forwback,libjnet), Void, (Cnet, Cmat, Cmat, Cint, Cint, Cint), net, x, y, length(net), size(x,2), batch)

function forward(net::Net, x::Mat, batch=100)
    y = similar(x, lsize(net[end],1), size(x,2))
    forward(net, x, y, batch)
    y
end

end # module Jnet
