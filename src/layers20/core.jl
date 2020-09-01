
abstract type Layer end

abstract type Loss <: Layer end

abstract type Activation <: Layer end

abstract type AbstractRNN{Params, Embed} <: Layer end

const LayerOrNothing = Union{Layer,Nothing}
const ActOrNothing   = Union{Activation,Nothing}
const DictOrNothing  = Union{Dict,Nothing}
const VVecOrNothing  = Union{Vector{Vector{<:Integer}},Nothing}
