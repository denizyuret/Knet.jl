# File I/O
using HDF5
using CUDArt

const h5xforw = [ noop, dropforw ]
const h5xback = [ noop, dropback ]
const h5yforw = [ noop, reluforw, softforw ]
const h5yback = [ noop, reluback, softback ]

function Layer(fname::String)
    f = h5open(fname, "r")
    l = Layer()
    l.w = read(f, "/w")
    l.b = read(f, "/b")
    xfunc = h5getattr1(f, "xfunc")
    yfunc = h5getattr1(f, "yfunc")
    l.xforw = h5xforw[1 + xfunc]
    l.xback = h5xback[1 + xfunc]
    l.yforw = h5yforw[1 + yfunc]
    l.yback = h5yback[1 + yfunc]
    close(f)
    l
end

function HDF5.h5write(fname::String, l::Layer)
    f = h5open(fname, "w")
    for n in names(l)
        if (isdefined(l,n)) 
            if (isa(l.(n), Array))
                f["/$n"] = l.(n)
            elseif (isa(l.(n), CudaArray))
                f["/$n"] = to_host(l.(n))
            end
        end
    end
    attrs(f)["xfunc"] = Int32[findfirst(h5xforw, l.xforw) - 1]
    attrs(f)["yfunc"] = Int32[findfirst(h5yforw, l.yforw) - 1]
    close(f)
end

function h5getattr1(f, name)
    a = attrs(f)
    if (!in(name, names(a)))
        warn("$(name) not defined, using 0.")
        return 0
    else
        return read(a[name])[1]
    end
end
