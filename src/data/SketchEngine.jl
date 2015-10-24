import Base: start, next, done

# file can be a Cmd, e.g. `xzcat sdfkl32KCsd_enTenTen12.vert.xz`

type SketchEngine; file; dict; unk; eostag; maxlen; column;
    function SketchEngine(file; dict=nothing, eostag="</s>", maxlen=40, column=1, o...)
        if dict == nothing
            unk = 0             # we will construct dict from data
            dict = Dict{Any,Int}(eostag=>eos)
        else
            isa(dict,Dict) || (dict = readvocab(dict))
            unk = length(dict)+1
        end
        new(file, dict, unk, eostag, maxlen, column)
    end
end

unk(s::SketchEngine)=s.unk
maxtoken(s::SketchEngine)=(s.unk > 0 ? s.unk : error("Please specify dict for maxtoken"))

start(s::SketchEngine)=open(s.file)
done(s::SketchEngine,io)=(isa(io,Tuple) && (io=io[1]); eof(io) && (close(io); true))

function next(s::SketchEngine, io)
    sent = Int32[]
    while true
        line = readline(isa(io,Tuple) ? io[1] : io)
        if startswith(line, s.eostag)
            if length(sent) <= s.maxlen
                break
            else
                empty!(sent)
            end
        elseif line[1]=='<' && line[2]!='\t'
            continue
        else
            token = split(line)[s.column]
            if s.unk == 0
                push!(sent, get!(s.dict, token, 1+length(s.dict)))
            else
                push!(sent, get(s.dict, token, s.unk))
            end
        end
    end
    return (sent, io)
end
