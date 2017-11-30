include("header.jl")

srand(42)
TOL=1e-1

# utils
std2(x) = let x_mu = x .- mean(x)
    mean(x_mu .* x_mu)
end

sizes = Dict([2=>(5,10), 4=>(3,4,5,3), 5=>(4,3,4,5,2)])
types = [Float32, Float64]
dims = [2, 4, 5]
# gradcheck functions
bn3(a) = batchnorm(a[1], nothing, a[2]; training=true)
bn1(a) = batchnorm(a; training=true)
bn3ts(a) = batchnorm(a[1], bnmoments(), a[2]; training=false)
bn1ts(a) = batchnorm(a, bnmoments(); training=false)
gpu_av = gpu() >= 0

@testset "batchnorm" begin
    for d in dims
        for et in types
            sz = sizes[d]
            tn(str) = string(str, d, "d")
            # CPU buffers
            ax = 5rand(et, sz...);
            C = sz[end-1]
            aw = bnparams(et, C)
            # GPU buffers
            kax = KnetArray{et}(5rand(sz...));
            kaw = KnetArray{et}(bnparams(C));

            # Array types
            at = typeof(ax)
            kat = typeof(kax)
            
            @testset "{$et, $d}" begin
                @testset "cpu-stat" begin
                    # The unit gaussian output
                    @test abs(mean(batchnorm(ax; training=true))) < 1e-3
                    @test isapprox(std2(batchnorm(ax; training=true)), 1.0; rtol=1e-2)
                end
                
                @testset "cpu-grads" begin
                    @test gradcheck(bn1, ax; rtol=TOL)
                    @test gradcheck(bn3, (ax, aw); rtol=TOL)
                end
                
                if gpu_av
                    @testset "gpu-stats" begin
                        # isapprox 0 didn't work
                        @test abs(mean(batchnorm(kax; training=true))) < 1e-3
                        @test isapprox(std2(batchnorm(kax; training=true)), 1.0; rtol=1e-2)
                    end

                    @testset "gpu-grads" begin
                        @test gradcheck(bn1, kax; rtol=TOL)
                        @test gradcheck(bn3, (kax, kaw); rtol=TOL)
                    end
                    
                    @testset "dev-consistency" begin
                        mc = bnmoments() #cpu
                        mg = bnmoments() #gpu
                        y1 = batchnorm(ax, mc)
                        y2 = batchnorm(kat(ax), mg) #use the same array
                        @test isapprox(mc.mean, at(mg.mean); rtol=TOL)
                        @test isapprox(mc.var, at(mg.var); rtol=TOL)
                        @test isapprox(y1, at(y2); rtol=TOL)
                    end
                end

                # Moments should be updates
                @testset "test-moments" begin
                    mc = bnmoments()
                    batchnorm(ax, mc)
                    m, v = copy(mc.mean), copy(mc.var)
                    batchnorm(.1ax.+3, mc)
                    @test isapprox(m, mc.mean; rtol=1e-20)
                    @test isapprox(v, mc.var; rtol=1e-20)
                    if gpu_av
                        mg = bnmoments()
                        batchnorm(kax, mg)
                        m, v = copy(mg.mean), copy(mg.var)
                        batchnorm(.1kax.+3, mg)
                        @test isapprox(m, mg.mean; rtol=1e-20)
                        @test isapprox(v, mg.var; rtol=1e-20)
                    end
                end
                
                @testset "training-moments" begin
                    mc = bnmoments()
                    batchnorm(ax, mc; training=true)
                    m, v = copy(mc.mean), copy(mc.var)
                    batchnorm(.1ax.+3, mc; training=true)
                    @test ~isapprox(m, mc.mean; rtol=1e-20)
                    @test ~isapprox(v, mc.var; rtol=1e-20)
                    if gpu_av
                        mg = bnmoments()
                        batchnorm(kax, mg; training=true)
                        m, v = copy(mg.mean), copy(mg.var)
                        batchnorm(.1kax.+3, mg; training=true)
                        @test ~isapprox(m, mg.mean; rtol=1e-20)
                        @test ~isapprox(v, mg.var; rtol=1e-20)
                    end 
                end
                
                if d > 2
                    @testset "cpu-grads-testing" begin
                        m1 = bnmoments()
                        @test gradcheck(bn1ts, ax; rtol=TOL)
                        @test gradcheck(bn3ts, (ax, aw); rtol=TOL)
                    end
                
                    if gpu_av
                        @testset "gpu-grads-testing" begin
                            @test gradcheck(bn1ts, kax; rtol=TOL)
                            @test gradcheck(bn3ts, (kax, kaw); rtol=TOL)
                        end
                    end
                end
                # TODO: add test mode gradchecks
            end #end of {dim, type} testset
        end
    end
end #end of testset

# suppress the return
nothing
