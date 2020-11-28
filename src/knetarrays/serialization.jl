# From @jbaron https://github.com/denizyuret/Knet.jl/issues/587

using Serialization

"""
Enable saving and loading of models by specialized KnetArray methods for Julia serialization
This will effectively move a GPU weight to the CPU before serializing it and move it back to
the GPU when deserializing.
"""
function Serialization.serialize(s::Serialization.AbstractSerializer, p::KnetArray)
    Serialization.serialize_type(s, typeof(p))
    Serialization.serialize(s, Array(p))
end

function Serialization.deserialize(s::Serialization.AbstractSerializer, t::Type{<:KnetArray})
    arr = Serialization.deserialize(s)
    return KnetArray(arr)
end
