#!/bin/env julia3

using HDF5

function main(file1, file2)
    f1 = h5open(file1, "r")
    f2 = h5open(file2, "r")
    gdiff(f1["/"], f2["/"])
    close(f1)
    close(f2)
end

function gdiff(o1, o2)
    for a in names(attrs(o1))
        if !in(a, names(attrs(o2)))
            # println("Only in $(filename(o1))[$(name(o1))$a]")
        end
    end
    for a in names(attrs(o2))
        if !in(a, names(attrs(o1)))
            # println("Only in $(filename(o2))[$(name(o2))$a]")
        else
            v1 = read(attrs(o1)[a])
            v2 = read(attrs(o2)[a])
            if v1 != v2
                println("Attr mismatch: $(filename(o1))[$(name(o1))$a]=$v1 $(filename(o2))[$(name(o2))$a]=$v2")
            end
        end
    end
    for a in names(o1)
        if !in(a, names(o2))
            # println("Only in $(filename(o1))[$(name(o1))$a]")
        end
    end
    for a in names(o2)
        if !in(a, names(o1))
            # println("Only in $(filename(o2))[$(name(o2))$a]")
        else
            c1 = o1[a]
            c2 = o2[a]
            if typeof(c1) != typeof(c2)
                println("Type mismatch: $(filename(c1))[$(name(c1))]=$(typeof(c1)) $(filename(c2))[$(name(c2))]=$(typeof(c2))")
            elseif isa(c1, HDF5Dataset)
                d1 = read(c1)
                d2 = read(c2)
                if size(d1) != size(d2)
                    println("Size mismatch: $(filename(c1))[$(name(c1))]=$(size(d1)) $(filename(c2))[$(name(c2))]=$(size(d2))")
                else
                    diff = maximum(abs(d1-d2))
                    println("maxdiff($(name(c1))) = $diff")
                end
            elseif isa(c1, HDF5Group)
                gdiff(c1, c2)
            else
                println("Don't know how to handle type $typeof(c1)")
            end
        end
    end
end

main(ARGS[1], ARGS[2])
