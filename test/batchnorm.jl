include("header.jl")

srand(42)
TOL=0.1

# CPU 4d buffers
ax4 = Array{Float64}(5rand(4,5,5,3));
ag4 = Array{Float64}(2randn(1,1,5,1));
ab4 = Array{Float64}(3randn(1,1,5,1));

# GPU 4d buffers
kax4 = KnetArray{Float64}(5rand(4,5,5,3));
kag4 = KnetArray{Float64}(2randn(1,1,5,1));
kab4 = KnetArray{Float64}(3randn(1,1,5,1));

# Array types
at = typeof(ax4)
kat = typeof(kax4)

# utils
stdev(x) = mean((x .- mean(x)) .* (x .- mean(x)))

# gradcheck functions
bn3(a) = batchnorm(a[1], a[2], a[3])
bn1(a) = batchnorm(a)

@testset "batchnorm" begin
    @testset "cpu-stats" begin
        # The unit gaussian output
        @test abs(mean(batchnorm(ax4))) < 1e-3
        @test isapprox(stdev(batchnorm(ax4)), 1.0; rtol=1e-2)
    end
    
    @testset "cpu-grads4d" begin
        @test gradcheck(bn1, ax4; rtol=TOL)
        @test gradcheck(bn3, (ag4, ab4, ax4); rtol=TOL)
    end
    
    if gpu() >= 0
        @testset "gpu-stats4d" begin
            # isapprox 0 didn't work
            @test abs(mean(batchnorm(kax4))) < 1e-3
            @test isapprox(stdev(batchnorm(kax4)), 1.0; rtol=1e-2)
        end

        @testset "gpu-grads4d" begin
            @test gradcheck(bn1, kax4; rtol=TOL)
            @test gradcheck(bn3, (kag4, kab4, kax4); rtol=TOL)
        end

        @testset "dev-consistency" begin
            mc = BNMoments() #cpu
            mg = BNMoments() #gpu
            y1 = batchnorm(mc, ax4)
            y2 = batchnorm(mg, kat(ax4)) #use the same array
            @test isapprox(mc.mean, at(mg.mean); rtol=TOL)
            @test isapprox(mc.var, at(mg.var); rtol=TOL)
            @test isapprox(y1, at(y2); rtol=TOL)
        end
    end

    # Moments should be updates
    @testset "test-moments" begin
        mc = BNMoments()
        batchnorm(mc, ax4)
        m, v = copy(mc.mean), copy(mc.var)
        batchnorm(mc, .1ax4.+3; training=false)
        @test isapprox(m, mc.mean; rtol=1e-20)
        @test isapprox(v, mc.var; rtol=1e-20)
        if gpu() >= 0
            mg = BNMoments()
            batchnorm(mg, kax4)
            m, v = copy(mg.mean), copy(mg.var)
            batchnorm(mg, .1kax4.+3; training=false)
            @test isapprox(m, mg.mean; rtol=1e-20)
            @test isapprox(v, mg.var; rtol=1e-20)
        end    
    end
    @testset "training-moments" begin
        mc = BNMoments()
        batchnorm(mc, ax4)
        m, v = copy(mc.mean), copy(mc.var)
        batchnorm(mc, .1ax4.+3)
        @test ~isapprox(m, mc.mean; rtol=1e-20)
        @test ~isapprox(v, mc.var; rtol=1e-20)
        if gpu() >= 0
            mg = BNMoments()
            batchnorm(mg, kax4)
            m, v = copy(mg.mean), copy(mg.var)
            batchnorm(mg, .1kax4.+3)
            @test ~isapprox(m, mg.mean; rtol=1e-20)
            @test ~isapprox(v, mg.var; rtol=1e-20)
        end 
    end
    
    # TODO: test other dimensionalities
end

# suppress the return
nothing
