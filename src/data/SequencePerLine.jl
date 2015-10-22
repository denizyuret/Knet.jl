type SequencePerLine; file; dict; unk; eos;
    function SequencePerLine(file; dict=nothing, o...)
        if dict == nothing
            unk = 0             # we will construct dict from data
            eos = 1
            dict = Dict{Any,Int}("</s>" => eos)
        else
            # unk = length(dict)+1
            # eos = length(dict)+2
            n = length(dict)    # DBG
            unk = get(dict, "<unk>", n+1)
            eos = get(dict, "<s>", unk==n+1 ? n+2 : n+1)
        end
        new(file, dict, unk, eos)
    end
end

unk(s::SequencePerLine)=s.unk
eos(s::SequencePerLine)=s.eos
maxtoken(s::SequencePerLine)=(s.unk > 0 ? max(s.unk,length(s.dict)) : error("Please specify dict for maxtoken"))

start(s::SequencePerLine)=open(s.file)
done(s::SequencePerLine,io)=(isa(io,Tuple) && (io=io[1]); eof(io) && (close(io); true))

function next(s::SequencePerLine, io)
    line = readline(isa(io,Tuple) ? io[1] : io)
    sent = Int32[]
    if s.unk == 0
        for token in split(line)
            push!(sent, get!(s.dict, token, 1+length(s.dict)))
        end
    else
        for token in split(line)
            push!(sent, get(s.dict, token, s.unk))
        end
    end
    return (sent, io)
end
