using Knet,BenchmarkTools

mul100(a,b)=(for i=1:100; c=a*b; end; Knet.@cuda(cudart,cudaDeviceSynchronize,()))

N = (10,100,1000)
T = (Float32,Float64)
for t in T, m in N, k in N, n in N
    info((t,m,k,n))
    a = KnetArray(rand(t, m, k))
    b = KnetArray(rand(t, k, n))
    r = @benchmarkable mul100($a,$b)
    s = min(1000,div(3e9,(100*m*n*sizeof(t))))
    println(run(r, samples=s))
    gpuinfo("before gc")
    a=b=nothing; gc(); sleep(1)
    gpuinfo("after  gc")
    knetgc()
    gpuinfo("after kgc")
end
