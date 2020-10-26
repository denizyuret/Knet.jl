using Printf
using Knet

"""struct Chain; layers; Chain(args...)= for i in args; new(i) end; end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)

struct Dense; w; b; f; end
(d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b)
Dense(i::Int,o::Int,f=relu) = Dense(param(o,i), param0(o), f);

struct Conv; w; b; f; end
(c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
Conv(w1,w2,cx,cy,f=relu) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f);

struct Linear; w; b; end

struct Embed; w; end
Embed(vocabsize::Int,embedsize::Int) = Embed(param(embedsize,vocabsize))
(e::Embed)(x) = e.w[:,x]"""

function model_summary(model)
    @printf("%s%35s%45s\n", "Layer Name", "Layer Size", "Activation Function")
    println(repeat("=",120))
    for i in model.layers
        @printf("%s%40s", typeof(i), size(i.w))
        try
            @printf("%40s\n", i.f)
        catch
            @printf("%40s\n", "Linear")
        end
    end
end


function Kf_cross_validation(model, data, set_size, opt_func; percent = false, epochs = 10, visualize = false)

end

"""
function create_model(layers)
    m = Any[]
    for i in layers
        if i[1] == "Dense"
            push!(m, Dense(i[2], i[3], if length(i) > 3 i[4]; else relu; end;))
        elseif i[1] == "Conv"
            push!(m, Conv(i[2], i[3],i[4], i[5], if length(i) > 5 i[6]; else ; end;))
        elseif i[1] == "Embed"
            push!(m, Embed(i[2], i[3]))
        end
    end
    return m[1:end]
end
"""

"""function add(model,unit_num; layer_name::string, f)

end


function add! (model,unit_num; layer_name::string, f)

end

function add(model,unit_num; layer_name::string, f)

end"""
