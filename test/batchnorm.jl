include("header.jl")

srand(42)
TOL=0.1

# CPU 4d buffers
ax4 = Array(5rand(4,5,5,3));
aw4 = bnparam(5)
# GPU 4d buffers
kax4 = KnetArray(5rand(4,5,5,3));
kaw4 = KnetArray(bnparam(5));

# Array types
at = typeof(ax4)
kat = typeof(kax4)

# utils
stdev(x) = mean((x .- mean(x)) .* (x .- mean(x)))

# gradcheck functions
bn3(a) = batchnorm(a[1], nothing, a[2]; training=true)
bn1(a) = batchnorm(a; training=true)

@testset "batchnorm" begin
    @testset "cpu-stats" begin
        # The unit gaussian output
        @test abs(mean(batchnorm(ax4; training=true))) < 1e-3
        @test isapprox(stdev(batchnorm(ax4; training=true)), 1.0; rtol=1e-2)
    end
    
    @testset "cpu-grads4d" begin
        @test gradcheck(bn1, ax4; rtol=TOL)
        @test gradcheck(bn3, (ax4, aw4); rtol=TOL)
    end
    
    if gpu() >= 0
        @testset "gpu-stats4d" begin
            # isapprox 0 didn't work
            @test abs(mean(batchnorm(kax4; training=true))) < 1e-3
            @test isapprox(stdev(batchnorm(kax4; training=true)), 1.0; rtol=1e-2)
        end

        @testset "gpu-grads4d" begin
            @test gradcheck(bn1, kax4; rtol=TOL)
            @test gradcheck(bn3, (kax4, kaw4); rtol=TOL)
        end

        @testset "dev-consistency" begin
            mc = BNMoments() #cpu
            mg = BNMoments() #gpu
            y1 = batchnorm(ax4, mc)
            y2 = batchnorm(kat(ax4), mg) #use the same array
            @test isapprox(mc.mean, at(mg.mean); rtol=TOL)
            @test isapprox(mc.var, at(mg.var); rtol=TOL)
            @test isapprox(y1, at(y2); rtol=TOL)
        end
    end

    # Moments should be updates
    @testset "test-moments" begin
        mc = BNMoments()
        batchnorm(ax4, mc)
        m, v = copy(mc.mean), copy(mc.var)
        batchnorm(.1ax4.+3, mc)
        @test isapprox(m, mc.mean; rtol=1e-20)
        @test isapprox(v, mc.var; rtol=1e-20)
        if gpu() >= 0
            mg = BNMoments()
            batchnorm(kax4, mg)
            m, v = copy(mg.mean), copy(mg.var)
            batchnorm(.1kax4.+3, mg)
            @test isapprox(m, mg.mean; rtol=1e-20)
            @test isapprox(v, mg.var; rtol=1e-20)
        end
    end
    @testset "training-moments" begin
        mc = BNMoments()
        batchnorm(ax4, mc; training=true)
        m, v = copy(mc.mean), copy(mc.var)
        batchnorm(.1ax4.+3, mc; training=true)
        @test ~isapprox(m, mc.mean; rtol=1e-20)
        @test ~isapprox(v, mc.var; rtol=1e-20)
        if gpu() >= 0
            mg = BNMoments()
            batchnorm(kax4, mg; training=true)
            m, v = copy(mg.mean), copy(mg.var)
            batchnorm(.1kax4.+3, mg; training=true)
            @test ~isapprox(m, mg.mean; rtol=1e-20)
            @test ~isapprox(v, mg.var; rtol=1e-20)
        end 
    end
    
    # TODO: test other dimensionalities
end

# suppress the return
nothing
