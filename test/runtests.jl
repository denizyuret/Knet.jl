#                            commit 8.3 8.3 6cb 6cb 8.2 6cb 6cb
#                           machine ai5 ai4 tr5 tr4 aws osx os4
@time include("distributions.jl") #   1   1   2   1   2   3   2
@time include("update.jl")        #  29  26 100  22 103  25  23   :: this was 30 at ai? slower if last? travis=100?
@time include("linalg.jl")        #  24  14  22   7  19  33  19
@time include("conv.jl")          #  22  12  62  47  34  44  16
