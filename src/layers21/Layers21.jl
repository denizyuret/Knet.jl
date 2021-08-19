module Layers21

include("init.jl")
include("block.jl")
include("add.jl")
include("mul.jl")
include("op.jl")
include("conv.jl")
include("linear.jl") # TODO: test, redesign
include("batchnorm.jl")
include("zeropad.jl")
include("show.jl")
include("jld2.jl")
include("embed.jl")

# TODO: rethink param/Param and generally parameter initialization, array_type etc.
#  xavier may not be optimal for embedding, should specify init options etc.
# TODO: rethink eliminating layers without parameters: dropout, activation

# 
# 
# include("dropout.jl")
# include("layernorm.jl")

end
