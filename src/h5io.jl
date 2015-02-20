# File I/O
using HDF5
using CUDArt

function Layer(fname::String)
    f = h5open(fname, "r")
    l = Layer()
    for n in names(f)
        s = symbol(n)
        if in(s, names(l))
            l.(s) = read(f, "/$n")
        end
    end
    for n in names(attrs(f))
        s = symbol(n)
        if in(s, names(l))
            l.(s) = eval(parse(read(attrs(f)[n])))
        end
    end
    close(f)
    return l
end

function HDF5.h5write(fname::String, l::Layer)
    f = h5open(fname, "w")
    for n in names(l)
        if (isdefined(l,n)) 
            if (isa(l.(n), Array))
                f["/$n"] = l.(n)
            elseif (isa(l.(n), CudaArray))
                f["/$n"] = to_host(l.(n))
            elseif (isa(l.(n), Function))
                attrs(f)[string(n)] = string(l.(n))
            end
        end
    end
    close(f)
end

