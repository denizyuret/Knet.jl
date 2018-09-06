using Pkg; for p in ("ZipFile","JLD2","FileIO"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using ZipFile, FileIO

# nltkurl has subdirectories like corpora, taggers, stemmers etc.
nltkurl = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages"
nltkdir = joinpath(@__DIR__, "nltk")

function brown()
    ispath(nltkdir) || mkpath(nltkdir)
    jldfile = joinpath(nltkdir,"brown.jld2")
    if !isfile(jldfile)
        file = joinpath(nltkdir,"brown.zip")
        isfile(file) || download("$nltkurl/corpora/brown.zip", file)
        r = ZipFile.Reader(file)
        data = []  # an array of (x,y) pairs where x is a word sequence, y is a tag sequence
        wdict = Dict{String,Int}()
        tdict = Dict{String,Int}()
        for f in r.files
            occursin(r"^brown/c.\d\d$", f.name) || continue
            for l in eachline(f)
                x = []; y = []
                for s in split(l)
                    t = findlast(isequal('/'), s)
                    word,tag = s[1:t-1],s[t+1:end] 
                    wdict[word] = 1 + get(wdict,word,0)
                    tdict[tag] = 1 + get(tdict,tag,0)
                    push!(x, word)
                    push!(y, tag)
                end
                isempty(x) || push!(data, (x,y))
            end
        end
        close(r)
        # Assign ids based on frequency
        words = Array{String}(undef, length(wdict)); tags = Array{String}(undef, length(tdict))
        n = 1; for (c,w) in sort(collect(zip(values(wdict),keys(wdict))), rev=true); wdict[w] = n; words[n] = w; n+=1; end
        n = 1; for (c,t) in sort(collect(zip(values(tdict),keys(tdict))), rev=true); tdict[t] = n; tags[n] = t; n+=1; end
        data = map(data) do xy; (UInt16[ wdict[x] for x in xy[1] ], UInt16[ tdict[y] for y in xy[2] ]); end
        return data,words,tags
        save(jldfile, "data", data, "words", words, "tags", tags)
    end
    load(jldfile, "data", "words", "tags")
end
