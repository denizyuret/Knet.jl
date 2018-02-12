"Where to download gutenberg from"
gutenbergurl = "http://www.gutenberg.org/files"

"Where to download gutenberg to"
gutenbergdir = Pkg.dir("Knet","data","gutenberg")

"Download text from Project Gutenberg and return contents as String."
function gutenberg(name)
    isdir(gutenbergdir) || mkpath(gutenbergdir)
    path = joinpath(gutenbergdir, "$name.txt")
    if !isfile(path)
        info("Downloading Gutenberg $name")
        url = "$gutenbergurl/$name/$name.txt"
        download(url,path)
    end
    return readstring(path)
end

"Filter and split Shakespeare text."
function shakespeare()
    global _shakespeare_trn, _shakespeare_tst, _shakespeare_chars
    if !isdefined(:_shakespeare_trn)
        s = gutenberg(100)
        a = []
        pos1 = 1
        while true
            pos2 = first(search(s, "<<THIS", pos1)) - 1
            if pos2 == -1
                push!(a, s[pos1:end])
                break
            end
            push!(a, s[pos1:pos2])
            pos1 = last(search(s, "SHIP.>>", pos2)) + 1
        end
        # 218 shakespeare texts
        a = a[3:end-1]
        # split trn, tst
        a = shuffle(MersenneTwister(42),a)
        trn = string(a[1:200]...)
        tst = string(a[201:end]...)
        # construct char vocab
        h = Dict{Char,Int}()
        for txt in (trn,tst), c in txt
            h[c] = 1 + get(h,c,0)
        end
        _shakespeare_chars = sort(collect(keys(h)),by=(x->h[x]),rev=true)
        _shakespeare_trn = UInt8[ findfirst(_shakespeare_chars,c) for c in trn ]
        _shakespeare_tst = UInt8[ findfirst(_shakespeare_chars,c) for c in tst ]
    end
    return _shakespeare_trn, _shakespeare_tst, _shakespeare_chars
end

nothing
