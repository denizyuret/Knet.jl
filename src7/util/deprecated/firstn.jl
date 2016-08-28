# TODO: use take and drop instead of firstn given in Julia base.

function firstn(a,n)
    b=Any[]
    for i in a
        push!(b,i)
        length(b)==n && break
    end
    b
end

function firstn(a::SketchEngine,n)
    D=length(a.dict)
    d=cell(D)
    for (w,i) in a.dict
        d[i]=w
    end
    b=Any[]
    for s in a
        push!(b,map(i->(i>D?"<unk>":d[i]), s))
        length(b)==n && break
    end
    b
end

function onerows(x::SparseMatrixCSC,dict::Dict)
    D=length(dict)+1
    d=cell(D)
    for (w,i) in dict
        d[i]=w
    end
    d[end]="<unk>"
    map(1:size(x,2)) do i
        x.nzval[i] == 0 ? "0" : d[x.rowval[i]]
    end
end

function onerows(x::SparseMatrixCSC)
    map(1:size(x,2)) do i
        x.nzval[i] == 0 ? 0 : x.rowval[i]
    end
end

function firstn(a::TagData,n)
    b=Any[]
    for i in a
        if i==nothing
            push!(b,i)
        else
            (x,y,m) = i
            !isa(a.x.sgen,Array) && isdefined(a.x.sgen,:dict) ?
            push!(b, (convert(Vector{Int},m), onerows(x,a.x.sgen.dict), onerows(y,a.y.sgen.dict))) :
            push!(b, map(z->convert(Vector{Int},z), (m, onerows(x), onerows(y))))
        end
        length(b)==n && break
    end
    b
end