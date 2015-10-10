function libsvm2sparse(fname)
    ncols = xrows = yrows = 0
    ydict = Dict{Int32,Int32}()
    info("Sizing $fname...")
    f = open(fname)
    for l in eachline(f)
        ncols += 1
        a = split(l)
        class = parse(Int32,a[1])
        ydict[class] = 1
        for i=2:length(a)
            aa = split(a[i],':')
            idx = parse(Int32,aa[1])
            idx > xrows && (xrows = idx)
        end
    end
    close(f)
    ykeys = sort(collect(keys(ydict)))
    yrows = length(ykeys)
    for i in 1:yrows
        ydict[ykeys[i]] = i
    end
    info("Found $ncols instances with xrows=$xrows yrows=$yrows")

    x = spzeros(Float32,Int32,xrows,ncols)
    y = zeros(Float32,yrows,ncols)

    info("Reading $fname...")
    f = open(fname)
    ncols = 0
    for l in eachline(f)
        ncols += 1
        a = split(l)
        class = parse(Int32,a[1])
        y[ydict[class], ncols] = 1
        for i=2:length(a)
            aa = split(a[i],':')
            idx = parse(Int32,aa[1])
            val = parse(Float32,aa[2])
            x[idx, ncols] = 1
        end
    end
    close(f)
    info("done.")
    return (x,y)
end
