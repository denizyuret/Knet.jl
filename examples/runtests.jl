using Knet, Base.Test
load_only = true
gcheck = 10

include("linreg.jl")
@time @show test1 = linreg("--gcheck $gcheck")
# 4.938186 seconds (5.33 M allocations: 293.787 MB, 1.52% gc time)
# 0.718748 seconds (371.98 k allocations: 70.803 MB, 1.21% gc time)
@test test1  == (0.0005497372347062405,32.77256166946498,0.11244349406523031)

include("mnist2d.jl")
@time @show test2 = mnist2d("--gcheck $gcheck")
# 8.949818 seconds (7.32 M allocations: 327.153 MB, 1.81% gc time)
# 6.499205 seconds (3.90 M allocations: 167.939 MB, 1.57% gc time)
@test test2  == (0.10628127f0,24.865438f0,3.5134742f0)

@time @show test3 = mnist2d("--ysparse --gcheck $gcheck")
# 8.470642 seconds (4.84 M allocations: 220.114 MB, 2.18% gc time)
# 7.720386 seconds (4.41 M allocations: 214.496 MB, 0.81% gc time)
@test test3  == (0.1062698f0,24.866688f0,3.513474f0)

@time @show test4 = mnist2d("--xsparse --gcheck $gcheck")
# 13.243564 seconds (5.11 M allocations: 802.085 MB, 1.82% gc time)
# 15.335449 seconds (4.88 M allocations: 756.636 MB, 1.27% gc time) after switching to sparse dw ???
# ---
# 12.380252 seconds (4.39 M allocations: 770.236 MB, 1.87% gc time) # with dw=CSR
# 14.695503 seconds (4.17 M allocations: 725.756 MB, 1.49% gc time) # with dw=CSRU: those atomic ops do have a cost
@test isapprox(test4[1], 0.10628127f0; rtol=0.005)
@test isapprox(test4[2], 24.865437f0; rtol=0.002)
@show isapprox(test4[3], 3.5134742f0; rtol=0.02) # cannot compute csru vecnorm

@time @show test5 = mnist2d("--xsparse --ysparse --gcheck $gcheck")
# 13.590826 seconds (5.14 M allocations: 839.147 MB, 1.24% gc time)
# 14.041564 seconds (4.68 M allocations: 794.398 MB, 2.26% gc time)
# 16.156115 seconds (4.49 M allocations: 750.289 MB, 1.63% gc time) after switching to sparse dw ???
# ---
# 13.390442 seconds (5.01 M allocations: 832.642 MB, 1.24% gc time)
# 13.959991 seconds (4.68 M allocations: 793.600 MB, 2.24% gc time)
@test isapprox(test5[1], 0.10628127f0; rtol=0.005)
@test isapprox(test5[2], 24.865437f0; rtol=0.002)
@show isapprox(test5[3], 3.5134742f0; rtol=0.02) # cannot compute csru vecnorm

include("mnist4d.jl")
@time @show test6 = mnist4d("--gcheck $gcheck")
# 18.867598 seconds (11.64 M allocations: 549.947 MB, 1.10% gc time)
# 16.554756 seconds (9.08 M allocations: 437.540 MB, 1.07% gc time)
@test isapprox(test6[1], 0.050180204f0; rtol=.01)
@test isapprox(test6[2], 25.783848f0;   rtol=.01)
@test isapprox(test6[3], 9.420588f0;    rtol=.1)

include("mnistpixels.jl")
# @time @show test7 = mnistpixels("--gcheck $gcheck")
# 9.909841 seconds (45.76 M allocations: 1.208 GB, 3.52% gc time)
# 8.877034 seconds (43.27 M allocations: 1.099 GB, 4.33% gc time)
# @test test7  == (0.1216,2.3023171f0,10.4108f0,30.598776f0)
# @test test7 == (0.12159999999999982,2.3023171f0,10.4108f0,30.598776f0) # switched to itembased
# @test test7 == (0.12159999999999982,2.3023171f0,10.412794f0,30.598776f0) # measuring wnorm after update now
@time @show test7 = mnistpixels("--gcheck 1 --nettype lstm --testfreq 2 --epochs 1 --batchsize 64 --epochsize 128") # switch to lstm so we can gradcheck, too slow for gcheck>1
@test test7 == (0,2.3025737f0,14.70776f0,0.12069904f0) # switched to --gcheck 1 --nettype lstm --testfreq 2 --epochs 1 --batchsize 64 --epochsize 128

include("adding.jl")
@time @show test8 = adding("--gcheck $gcheck --epochs 1 --nettype lstm")
# 9.207238 seconds (17.03 M allocations: 738.786 MB, 2.00% gc time)
# 9.114330 seconds (16.23 M allocations: 704.629 MB, 1.80% gc time)
# @test test8  == (0.04885713f0, 5.6036315f0,3.805253f0) 
# @test test8  == (0.04885713f0, 5.6057444f0, 3.805253f0) # measuring wnorm after update now
# @test test8 == (0.05627571f0,5.484082f0,4.1594324f0) # new generator
@test test8 == (0.24768005f0,3.601481f0,1.2290705f0) # switched to --epochs 1 --nettype lstm, gradcheck does not work with irnn/relu

include("rnnlm.jl")
@time @show test9 = rnnlm("ptb.valid.txt ptb.test.txt --gcheck $gcheck")
# 32.368835 seconds (22.35 M allocations: 2.210 GB, 1.56% gc time)
# 22.892147 seconds (22.46 M allocations: 945.257 MB, 2.17% gc time) after switching to Float32
# ---
# 21.215483 seconds (20.45 M allocations: 861.323 MB, 4.25% gc time)

# This is for: Float64
# @test isapprox(test9[1], 814.9780887272417;  rtol=.0001)
# @test isapprox(test9[2], 541.2457922913605;  rtol=.0001)
# @test isapprox(test9[3], 267.626257438979;   rtol=.005)
# @test isapprox(test9[4], 120.16170771885587; rtol=.0001)

# Changing to: Float32
@test isapprox(test9[1], 825.336, rtol=0.05)
@test isapprox(test9[2], 531.640, rtol=0.05)
@test isapprox(test9[3], 267.337, rtol=.005)
@test isapprox(test9[4], 136.923, rtol=0.0001)

include("copyseq.jl")
@time @show test10 = copyseq("--getloss --getnorm --epochs 1 --gcheck $gcheck ptb.valid.txt ptb.test.txt")
# 15.756479 seconds (18.23 M allocations: 803.964 MB, 2.30% gc time)
# 6.381941 seconds (6.52 M allocations: 276.747 MB, 2.82% gc time)
@test isapprox(test10[1], 4269.43; rtol=.0001)
@test isapprox(test10[2], 1378.93; rtol=.0001)
@test isapprox(test10[3], 104.891; rtol=.0001)
@test isapprox(test10[4], 256.03;  rtol=.0001)

