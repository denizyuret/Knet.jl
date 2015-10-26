using Knet, ArgParse, Base.Test
load_only = true

s = ArgParseSettings()
@add_arg_table s begin
    ("--all"; action=:store_true)
    ("--linreg"; action=:store_true)
    ("--mnist2d"; action=:store_true)
    ("--mnist2dy"; action=:store_true)
    ("--mnist2dx"; action=:store_true)
    ("--mnist2dxy"; action=:store_true)
    ("--mnist4d"; action=:store_true)
    ("--mnistpixels"; action=:store_true)
    ("--adding"; action=:store_true)
    ("--rnnlm"; action=:store_true)
    ("--copyseq"; action=:store_true)
    ("--twice"; action=:store_true)
    ("--gcheck"; arg_type=Int; default=0)
end
opts = parse_args(ARGS, s)
gcheck = opts["gcheck"]
twice = opts["twice"]

if opts["all"] || opts["linreg"]
    include("linreg.jl")
    @time @show test1 = linreg("--gcheck $gcheck")
    #@test test1 == (0.0005497372347062405,32.77256166946498,0.11244349406523031)
    @test  test1 == (0.0005497372347062409,32.77256166946497,0.11244349406522969) # Mon Oct 26 11:10:17 PDT 2015: update uses axpy to scale with gclip&lr
    twice && (@time @show test1 = linreg("--gcheck $gcheck"))
    # 0.739858 seconds (394.09 k allocations: 71.335 MB, 1.23% gc time) Tue Oct 20 18:29:41 PDT 2015
end

if opts["all"] || opts["mnist2d"]
    include("mnist2d.jl")
    @time @show test2 = mnist2d("--gcheck $gcheck")
    # @test test2 == (0.10628127f0,24.865438f0,3.5134742f0)
    @test test2 == (0.10626979f0,24.866688f0,3.5134728f0) # softloss with mask
    twice && (@time @show test2 = mnist2d("--gcheck $gcheck"))
    # 6.941715 seconds (3.35 M allocations: 151.876 MB, 1.33% gc time) Tue Oct 20 19:15:59 PDT 2015
    # 6.741272 seconds (3.35 M allocations: 151.858 MB, 1.41% gc time) # Mon Oct 26 11:10:17 PDT 2015: update uses axpy to scale with gclip&lr
end

if opts["all"] || opts["mnist2dy"]
    isdefined(:mnist2d) || include("mnist2d.jl")
    @time @show test3 = mnist2d("--ysparse --gcheck $gcheck")
    @test test3  == (0.1062698f0,24.866688f0,3.513474f0)
    twice && (@time @show test3 = mnist2d("--ysparse --gcheck $gcheck"))
    # 8.478264 seconds (3.59 M allocations: 173.689 MB, 2.06% gc time) Tue Oct 20 19:14:45 PDT 2015
    # 8.205758 seconds (3.59 M allocations: 173.636 MB, 2.14% gc time) # Mon Oct 26 11:10:17 PDT 2015: update uses axpy to scale with gclip&lr
end

if opts["all"] || opts["mnist2dx"]
    isdefined(:mnist2d) || include("mnist2d.jl")
    @time @show test4 = mnist2d("--xsparse --gcheck $gcheck")

    @test isapprox(test4[1], 0.10628127f0; rtol=0.005)
    @test isapprox(test4[2], 24.865437f0; rtol=0.002)
    @test isapprox(test4[3], 3.5134742f0; rtol=0.02) # cannot compute csru vecnorm

    twice && (@time @show test4 = mnist2d("--xsparse --gcheck $gcheck"))
    # 12.362125 seconds (3.81 M allocations: 753.744 MB, 1.87% gc time)  Tue Oct 20 19:13:25 PDT 2015
    # 11.751002 seconds (3.84 M allocations: 753.959 MB, 1.95% gc time) # Mon Oct 26 11:10:17 PDT 2015: update uses axpy to scale with gclip&lr
end

if opts["all"] || opts["mnist2dxy"]
    isdefined(:mnist2d) || include("mnist2d.jl")
    @time @show test5 = mnist2d("--xsparse --ysparse --gcheck $gcheck")
    @test isapprox(test5[1], 0.10628127f0; rtol=0.005)
    @test isapprox(test5[2], 24.865437f0; rtol=0.002)
    @test isapprox(test5[3], 3.5134742f0; rtol=0.02) # cannot compute csru vecnorm

    twice && (@time @show test5 = mnist2d("--xsparse --ysparse --gcheck $gcheck"))
    # 14.077099 seconds (4.09 M allocations: 776.263 MB, 2.22% gc time) Tue Oct 20 19:11:52 PDT 2015
    # 13.320959 seconds (4.11 M allocations: 776.397 MB, 2.29% gc time) # Mon Oct 26 11:10:17 PDT 2015: update uses axpy to scale with gclip&lr
end

if opts["all"] || opts["mnist4d"]
    include("mnist4d.jl")
    @time @show test6 = mnist4d("--gcheck $gcheck")

    @test isapprox(test6[1], 0.050180204f0; rtol=.01)
    @test isapprox(test6[2], 25.783848f0;   rtol=.01)
    @test isapprox(test6[3], 9.420588f0;    rtol=.1)

    twice && (@time @show test6 = mnist4d("--gcheck $gcheck"))
    # 17.093371 seconds (10.15 M allocations: 479.611 MB, 1.11% gc time)  Tue Oct 20 19:09:19 PDT 2015
end

if opts["all"] || opts["mnistpixels"]
    include("mnistpixels.jl")

    # @time @show test7 = mnistpixels("--gcheck $gcheck")
    # 9.909841 seconds (45.76 M allocations: 1.208 GB, 3.52% gc time)
    # 8.877034 seconds (43.27 M allocations: 1.099 GB, 4.33% gc time)
    # @test test7  == (0.1216,2.3023171f0,10.4108f0,30.598776f0)
    # @test test7 == (0.12159999999999982,2.3023171f0,10.4108f0,30.598776f0) # switched to itembased
    # @test test7 == (0.12159999999999982,2.3023171f0,10.412794f0,30.598776f0) # measuring wnorm after update now

    # switch to lstm so we can gradcheck, too slow for gcheck>1
    @time @show test7 = mnistpixels("--gcheck $gcheck --nettype lstm --testfreq 2 --epochs 1 --batchsize 64 --epochsize 128") 
    @test test7 == (0,2.3025737f0,14.70776f0,0.12069904f0) # switched to --gcheck 1 --nettype lstm --testfreq 2 --epochs 1 --batchsize 64 --epochsize 128
    twice && (@time @show test7 = mnistpixels("--gcheck $gcheck --nettype lstm --testfreq 2 --epochs 1 --batchsize 64 --epochsize 128"))
    # 2.599979 seconds (5.19 M allocations: 212.248 MB, 2.77% gc time)  Tue Oct 20 19:07:11 PDT 2015
end

if opts["all"] || opts["adding"]
    include("adding.jl")
    @time @show test8 = adding("--gcheck $gcheck --epochs 1 --nettype lstm")

    # @test test8  == (0.04885713f0, 5.6036315f0,3.805253f0)  # --epochs 20 --nettype irnn
    # @test test8  == (0.04885713f0, 5.6057444f0, 3.805253f0) # measuring wnorm after update now
    # @test test8 == (0.05627571f0,5.484082f0,4.1594324f0)    # new generator
    @test test8 == (0.24768005f0,3.601481f0,1.2290705f0)      # switched to --epochs 1 --nettype lstm, gradcheck does not work with irnn/relu

    twice && (@time @show test8 = adding("--gcheck $gcheck --epochs 1 --nettype lstm"))
    # 9.114330 seconds (16.23 M allocations: 704.629 MB, 1.80% gc time) # --epochs 20 --nettype irnn
    # 2.028728 seconds (3.82 M allocations: 164.958 MB, 1.95% gc time)  Tue Oct 20 19:03:01 PDT 2015
end

if opts["all"] || opts["rnnlm"]
    include("rnnlm.jl")
    if !isfile("ptb.valid.txt")
        info("Downloading ptb...")
	run(pipeline(`wget -q -O- http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz`,
                     `tar --strip-components 3 -xvzf - ./simple-examples/data/ptb.valid.txt ./simple-examples/data/ptb.test.txt`))
    end
    @time @show test9 = rnnlm("ptb.valid.txt ptb.test.txt --gcheck $gcheck")

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

    twice && (@time @show test9 = rnnlm("ptb.valid.txt ptb.test.txt --gcheck $gcheck"))
    # 32.368835 seconds (22.35 M allocations: 2.210 GB, 1.56% gc time)   for Float64
    # 22.892147 seconds (22.46 M allocations: 945.257 MB, 2.17% gc time) after switching to Float32
    # 21.982870 seconds (20.64 M allocations: 866.929 MB, 3.08% gc time) Tue Oct 20 19:00:29 PDT 2015
end

if opts["all"] || opts["copyseq"]
    if !isfile("ptb.valid.txt")
        info("Downloading ptb...")
	run(pipeline(`wget -q -O- http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz`,
                     `tar --strip-components 3 -xvzf - ./simple-examples/data/ptb.valid.txt ./simple-examples/data/ptb.test.txt`))
    end
    include("copyseq.jl")
    @time @show test10 = copyseq("--epochs 1 --gcheck $gcheck ptb.valid.txt ptb.test.txt")
    @test isapprox(test10[1], 3143.22; rtol=.0001)
    @test isapprox(test10[2], 1261.19; rtol=.0001)
    @test isapprox(test10[3], 106.760; rtol=.0001)
    @test isapprox(test10[4], 206.272; rtol=.0001)
    twice && (@time @show test10 = copyseq("--epochs 1 --gcheck $gcheck ptb.valid.txt ptb.test.txt"))
    # 5.984980 seconds (8.33 M allocations: 353.611 MB, 4.15% gc time) Tue Oct 20 18:58:25 PDT 2015
    # 11.230476 seconds (16.29 M allocations: 701.612 MB, 4.05% gc time) Wed Oct 21 23:19:24 PDT 2015 (unsorted input)
end
