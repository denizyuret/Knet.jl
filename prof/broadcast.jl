using BenchmarkTools,Knet
sizes = (1,10,100,1000)
for r in (0,1,2)
    println(r==0 ? "a[m,n].+b" : r==1 ? "a[m,n].+b[1,n]" : "a[m,n].+b[m,1]")
    for s in sizes; print("\t$s"); end; println()
    for nrows in sizes
        print(nrows)
        for ncols in sizes
            a = KnetArray(rand(Float32,nrows,ncols))
            b = (r==0 ? rand() : r==1 ? KnetArray(rand(Float32,1,ncols)) : KnetArray(rand(Float32,nrows,1)))
            c = @benchmark ($a.+$b) seconds=1
            m = round(Int, mean(c.times))
            print("\t$m")
        end
        println()
    end
end

