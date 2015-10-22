# file can be a Cmd, e.g. `xzcat sdfkl32KCsd_enTenTen12.vert.xz`

type SketchEngine; file; dict; eos; unk; eostag;
    function SketchEngine(file; dict=nothing, eostag="</s>", o...)
        if dict == nothing
            unk = 0             # we will construct dict from data
            eos = 1
            dict = Dict{Any,Int}(eostag=>eos)
        else
            unk = length(dict)+1
            eos = length(dict)+2
        end
        new(file, dict, eos, unk, eostag)
    end
end

unk(s::SketchEngine)=s.unk
eos(s::SketchEngine)=s.eos
maxtoken(s::SketchEngine)=(s.unk > 0 ? s.unk : error("Please specify dict for maxtoken"))

start(s::SketchEngine)=open(s.file)
done(s::SketchEngine,io)=(isa(io,Tuple) && (io=io[1]); eof(io) && (close(io); true))

function next(s::SketchEngine, io)
    sent = Int32[]
    while true
        line = readline(isa(io,Tuple) ? io[1] : io)
        if startswith(line, s.eostag)
            break
        elseif line[1]=='<' && line[2]!='\t'
            continue
        else
            token = split(line)[1]
            if s.unk == 0
                push!(sent, get!(s.dict, token, 1+length(s.dict)))
            else
                push!(sent, get(s.dict, token, s.unk))
            end
        end
    end
    return (sent, io)
end
