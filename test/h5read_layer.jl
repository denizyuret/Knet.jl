using HDF5
include("../julia/kunet.jl")

xforw = [ KUnet.noop, KUnet.dropforw ]
xback = [ KUnet.noop, KUnet.dropback ]
yforw = [ KUnet.noop, KUnet.reluforw, KUnet.softforw ]
yback = [ KUnet.noop, KUnet.reluback, KUnet.softback ]

function h5read_layer(fname)
    f = h5open(fname, "r")
    l = KUnet.Layer()
    l.w = read(f, "/w")
    l.b = read(f, "/b")
    xfunc = h5getattr1(f, "xfunc")
    yfunc = h5getattr1(f, "yfunc")
    l.xforw = xforw[1 + xfunc]
    l.xback = xback[1 + xfunc]
    l.yforw = yforw[1 + yfunc]
    l.yback = yback[1 + yfunc]
    close(f)
    l
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
