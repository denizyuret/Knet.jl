import JLD.save

function save(filename::String, net::Net)
    net = map(layer2jld, net)
    save(filename, "net", net)
end

function layer2jld(l::Layer)
    l = copy(l,:cpu)
    isdefined(l,:f) && (l.f = string(l.f))
    isdefined(l,:fx) && (l.fx = string(l.fx))
    return l
end

function newnet(filename::String)
    net = load(filename, "net")
    map(jld2layer, net)
end

function jld2layer(l::Layer)
    isdefined(l,:f) && (l.f = eval(parse(l.f)))
    isdefined(l,:fx) && (l.fx = eval(parse(l.fx)))
    GPU && (l = copy(l,:gpu))
    return l
end
