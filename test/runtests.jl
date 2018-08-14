#                            commit 9.2 8.3 8.3 6cb 6cb 8.6 6cb 6cb
#                           machine tig ai5 ai4 tr5 tr4 aws osx os4
@time include("kptr.jl")          #  16   1   1   0   0  20   0   0
@time include("gpu.jl")           #   6   1   1   0   0   2   0   0
@time include("distributions.jl") #   2   1   1   2   1   3   3   2
@time include("dropout.jl")       #   5                   2        
@time include("loss.jl")          #  10                   4        
@time include("rnn.jl")           #  81                  12        
@time include("karray.jl")        #  55  19  12   -   -  21   -   0
@time include("update.jl")        #  61  29  26 100  22  72  25  23 
@time include("conv.jl")          # 107  22  12  62  47  26  44  16
@time include("linalg.jl")        #  62  24  14  22   7  28  33  19
@time include("broadcast.jl")     #  56  34  19 491 119  51  53  25
@time include("unary.jl")         # 122  42   6  36   4  56  67  11
@time include("reduction.jl")     # 106  40  21  29  11  57  55  29
@time include("batchnorm.jl")     #  93                            
