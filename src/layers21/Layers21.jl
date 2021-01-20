module Layers21

include("init.jl")
include("batchnorm.jl")
include("conv.jl")

# TODO: rethink param/Param and generally parameter initialization, array_type etc.
#  xavier may not be optimal for embedding, should specify init options etc.
# TODO: rethink eliminating layers without parameters: dropout, activation

# 
# include("dense.jl")
# include("dropout.jl")
# include("embed.jl")
# include("layernorm.jl")

end
