import HDF5: h5write, h5read
export h5write
const dont_save = [:y, :x, :dx, :xdrop]

function Layer(fname::String)
    f = h5open(fname, "r")
    l = Layer()
    h5read(f["/"], l)
    close(f)
    return l
end

function h5read(g::HDF5Group, l::Union(Layer,UpdateParam))
    a = attrs(g)
    for n in names(a)
        s = symbol(n)
        if in(s, names(l))
            l.(s) = read(a[n])
            isa(l.(s), String) && (l.(s) = eval(parse(l.(s))))
        else
            warn("$n is not a valid field")
        end
    end
    for n in names(g)
        s = symbol(n)
        if in(s, names(l))
            if isa(g[n], HDF5Dataset)
                l.(s) = read(g[n])
                usegpu && (l.(s) = CudaArray(l.(s)))
            elseif isa(g[n], HDF5Group)
                assert(isa(l, Layer))
                l.(s) = UpdateParam()
                h5read(g[n], l.(s))
            else
                warn("Don't know how to read $n::$(typeof(g[n]))")
            end
        else
            warn("$n is not a valid field")
        end
    end
end

function h5write(fname::String, l::Layer)
    f = h5open(fname, "w")
    h5write(f["/"], l)
    close(f)
end

function h5write(g::HDF5Group, l)
    for n in names(l)
        if (isdefined(l,n) && !in(n, dont_save))
            if (isa(l.(n), Array))
                g["$n"] = l.(n)
            elseif (isdefined(:CudaArray) && isa(l.(n), CudaArray))
                g["$n"] = to_host(l.(n))
            elseif (isa(l.(n), Function))
                attrs(g)[string(n)] = string(l.(n))
            elseif (isa(l.(n), Number))
                attrs(g)[string(n)] = l.(n)
            elseif (isa(l.(n), UpdateParam))
                gg = g_create(g, string(n))
                h5write(gg, l.(n))
            else
                error("Don't know how to save $(typeof(l.(n)))::$n")
            end
        end
    end
end
