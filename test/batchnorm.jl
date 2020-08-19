using Test, Statistics
using Knet.Ops20: batchnorm, bnmoments, bnparams
using Knet.KnetArrays: KnetArray
using AutoGrad: gradcheck
using CUDA: CUDA, functional

@testset "batchnorm" begin

    # Random.seed!(42)

    # utils
    std2(x) = let x_mu = x .- mean(x)
        mean(x_mu .* x_mu)
    end

    # gradcheck functions
    bn3(a) = batchnorm(a[1], nothing, a[2]; training=true)
    bn1(a) = batchnorm(a; training=true)
    bn3ts(a) = batchnorm(a[1], bnmoments(), a[2]; training=false)
    bn1ts(a) = batchnorm(a, bnmoments(); training=false)

    types = (Float64,) #TODO: [Float32, Float64]
    sizes = Dict([2=>(5,10), 4=>(3,4,5,3), 5=>(4,3,4,5,2)])
    dims = [2, 4, 5]
    gpu_av = CUDA.functional()

    for d in dims
        for et in types
            sz = sizes[d]
            tn(str) = string(str, d, "d")
            # CPU buffers
            ax = 5rand(et, sz...);
            C = sz[end-1]
            aw = bnparams(et, C)
            # GPU buffers
            kax = gpu_av ? KnetArray{et}(5rand(sz...)) : nothing
            kaw = gpu_av ? KnetArray{et}(bnparams(C)) : nothing

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
                    @test gradcheck(bn1, ax)
                    @test gradcheck(bn3, (ax, aw))
                end
                
                if gpu_av
                    @testset "gpu-stats" begin
                        # isapprox 0 didn't work
                        @test abs(mean(batchnorm(kax; training=true))) < 1e-3
                        @test isapprox(std2(batchnorm(kax; training=true)), 1.0; rtol=1e-2)
                    end

                    @testset "gpu-grads" begin
                        @test gradcheck(bn1, kax)
                        @test gradcheck(bn3, (kax, kaw))
                    end
                    
                    @testset "dev-consistency" begin
                        mc = bnmoments() #cpu
                        mg = bnmoments() #gpu
                        y1 = batchnorm(ax, mc)
                        y2 = batchnorm(kat(ax), mg) #use the same array
                        @test isapprox(mc.mean, at(mg.mean))
                        @test isapprox(mc.var, at(mg.var))
                        @test isapprox(y1, at(y2))
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
                # TODO: Remove this if after 2d mode test backward supported
                if d > 2
                    @testset "cpu-grads-testing" begin
                        m1 = bnmoments()
                        @test gradcheck(bn1ts, ax)
                        @test gradcheck(bn3ts, (ax, aw))
                    end
                
                    if gpu_av
                        @testset "gpu-grads-testing" begin
                            @test gradcheck(bn1ts, kax)
                            @test gradcheck(bn3ts, (kax, kaw))
                        end
                    end
                end
            end #end of {dim, type} testset
        end
    end
end #end of testset

# suppress the return
nothing
