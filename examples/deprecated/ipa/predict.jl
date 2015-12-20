using Knet,JLD,ArgParse
using Knet: readvocab, lookup

function main(args=ARGS)
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--modelfile"; required=true)
        ("--dictfiles"; nargs=2; required=true)
        ("--beamsize"; arg_type=Int; default=16)
        ("--nbest"; arg_type=Int; default=1)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args, s)
    model = gpucopy(load(opts["modelfile"],"model"))
    dict1 = readvocab(opts["dictfiles"][1])
    dict2 = readvocab(opts["dictfiles"][2])
    for l in eachline(STDIN)
        word = map(c->dict1[c], split(l))
        pred = predict(model, Any[word]; beamsize=opts["beamsize"], nbest=opts["nbest"], dense=true)
        pstr = lookup(pred[1], dict2)
        println(join(map(x->join(x," "), pstr), "\t"))
    end
end

!isinteractive() && !isdefined(:load_only) && main(ARGS)
