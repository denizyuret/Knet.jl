function Layer(w; args...)
    l = Layer()
    l.w = w
    for (k,v)=args 
        if in(k, names(l)) 
            l.(k) = v
        else 
            warn("Layer has no field $k")
        end
    end
    return l
end
