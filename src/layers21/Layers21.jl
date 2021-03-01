module Layers21

include("init.jl")
include("sequential.jl")
include("residual.jl")
include("squeezeexcitation.jl")
include("batchnorm.jl")
include("conv.jl")
include("linear.jl") # TODO: test, redesign
include("op.jl")
include("zeropad.jl")
include("show.jl")
include("jld2.jl")

# TODO: rethink param/Param and generally parameter initialization, array_type etc.
#  xavier may not be optimal for embedding, should specify init options etc.
# TODO: rethink eliminating layers without parameters: dropout, activation

# 
# 
# include("dropout.jl")
# include("embed.jl")
# include("layernorm.jl")

end
