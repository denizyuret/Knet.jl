using Knet, Test
macro timeinclude(x)
    :(print($x); print("\t"); @time include($x))
end

@testset "Knet" begin

#                            commit ee1 c6a 0b5 e3a 9.2 8.3 8.3 6cb 6cb 8.6 6cb 6cb
#                           machine  ci rzr  sc  sc tig ai5 ai4 tr5 tr4 aws osx os4
@timeinclude("kptr.jl")          #   ?   4   3   8  16   1   1   0   0  20   0   0
@timeinclude("gpu.jl")           #   3   1   2   2   6   1   1   0   0   2   0   0
@timeinclude("distributions.jl") #   1   1   1   1   2   1   1   2   1   3   3   2
@timeinclude("dropout.jl")       #   6   1   2   8   5                   2
@timeinclude("serialize.jl")     #  16   2  11   1
@timeinclude("jld.jl")           #  11  11   9  26
@timeinclude("gcnode.jl")
@timeinclude("statistics.jl")    #
@timeinclude("cuarray.jl")       #  25
@timeinclude("bmm.jl")           #  10   3   9 
@timeinclude("linalg.jl")        #  33  17  22  22  62  24  14  22   7  28  33  19
@timeinclude("batchnorm.jl")     #  32  14  23  22  93
@timeinclude("loss.jl")          #  32  13  19  20  10                   4
@timeinclude("karray.jl")        #  49  20  27  21  55  19  12   -   -  21   -   0
@timeinclude("reduction.jl")     #  67  37  52  55 106  40  21  29  11  57  55  29
@timeinclude("conv.jl")          #  71  33  48  51 107  22  12  62  47  26  44  16
@timeinclude("rnn.jl")           #  95  52  54  41  81                  12
@timeinclude("binary.jl")        # 124  64  75  80  56  34  19 491 119  51  53  25
@timeinclude("unary.jl")         # 172  86 103 103 122  42   6  36   4  56  67  11
@timeinclude("update.jl")        # 316  61  82  60  61  29  26 100  22  72  25  23
#TODO include("data.jl")
#TODO include("hyperopt.jl")
#TODO include("progress.jl")
#TODO include("random.jl")
#TODO include("train.jl")
#TODO include("uva.jl")

end
