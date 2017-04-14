using BenchmarkTools,Knet
sizes = (1,10,100,1000)
for region in ((1,2),(1,),(2,))
    println("sum(a,$region)")
    for s in sizes; print("\t$s"); end; println()
    for nrows in sizes
        print(nrows)
        for ncols in sizes
            a = KnetArray(rand(Float32,nrows,ncols))
            b = @benchmark sum($a,$region) seconds=1
            m = round(Int, mean(b.times))
            print("\t$m")
        end
        println()
    end
end

