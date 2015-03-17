import JLD.save

function save(filename::AbstractString, net::Net)
    net = map(savelayer, net)
    save(filename, "net", net)
end

function savelayer(l::Layer)
    l = copy(l,:cpu)
    isdefined(l,:f) && (l.f = symbol(l.f))
    isdefined(l,:fx) && (l.fx = symbol(l.fx))
    return l
end

function newnet(filename::AbstractString)
    net = load(filename, "net")
    map(loadlayer, net)
end

function loadlayer(l::Layer)
    isdefined(l,:f) && (l.f = eval(l.f))
    isdefined(l,:fx) && (l.fx = eval(l.fx))
    usegpu && (l = copy(l,:gpu))
    return l
end
