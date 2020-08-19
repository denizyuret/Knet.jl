using Test

macro timeinclude(x)
    :(print($x); print("\t"); @time include($x))
end

@testset "Knet" begin

#                            commit 555 686 ee1 c6a 0b5 e3a 9.2 8.3 8.3 6cb 6cb 8.6 6cb 6cb
#                           machine dy3 rzr  ci rzr  sc  sc tig ai5 ai4 tr5 tr4 aws osx os4
@timeinclude("kptr.jl")          #   30  2   ?   4   3   8  16   1   1   0   0  20   0   0
@timeinclude("gpu.jl")           #   31  2   3   1   2   2   6   1   1   0   0   2   0   0
@timeinclude("distributions.jl") #   26  1   1   1   1   1   2   1   1   2   1   3   3   2
@timeinclude("dropout.jl")       #   31  3   6   1   2   8   5                   2
@timeinclude("gcnode.jl")        #   39  1
@timeinclude("jld.jl")           #   45  2  11  11   9  26
@timeinclude("statistics.jl")    #   48  7
@timeinclude("bmm.jl")           #   65  9  10   3   9 
@timeinclude("serialize.jl")     #   70 20  16   2  11   1
@timeinclude("loss.jl")          #   74 17  32  13  19  20  10                   4
@timeinclude("cuarray.jl")       #   74 18  25
@timeinclude("update.jl")        #   78 64 316  61  82  60  61  29  26 100  22  72  25  23
@timeinclude("linalg.jl")        #   85 17  33  17  22  22  62  24  14  22   7  28  33  19
@timeinclude("batchnorm.jl")     #   92 19  32  14  23  22  93
@timeinclude("ops20.jl")         #   94
@timeinclude("karray.jl")        #  114 27  49  20  27  21  55  19  12   -   -  21   -   0
@timeinclude("conv.jl")          #  114 47  71  33  48  51 107  22  12  62  47  26  44  16
@timeinclude("rnn.jl")           #  126 43  95  52  54  41  81                  12
@timeinclude("reduction.jl")     #  137 42  67  37  52  55 106  40  21  29  11  57  55  29
@timeinclude("unary.jl")         #  174 76 172  86 103 103 122  42   6  36   4  56  67  11
@timeinclude("binary.jl")        #  234 89 124  64  75  80  56  34  19 491 119  51  53  25
#TODO include("data.jl")
#TODO include("hyperopt.jl")
#TODO include("progress.jl")
#TODO include("random.jl")
#TODO include("train.jl")
#TODO include("uva.jl")

end
