"""
TagData(data1, data2; batch=128, ftype=Float32, dense=false) creates a
data generator that can be used with a Tagger model.  The source data1
and target data2 should be sequence generators, i.e. next(data1)
should deliver a vector of Ints that represent the next sequence.
This division of labor allows different file formats to be supported.
maxtoken(data1) should give the largest integer produced by data1.

The following transformations are performed by an TagData generator:

* sequences are minibatched according to the batch argument.
* sequences in a minibatch padded to all be the same length.
* the end-of-sequence is indicated using a "nothing" token.

"""
type TagData; x; y; mask;
    function TagData(xgen, ygen; batchsize=128, ftype=Float32, dense=false, o...)
        new(TagBatch(xgen, batchsize, ftype, dense),
            TagBatch(ygen, batchsize, ftype, dense),
            zeros(Cuchar, batchsize))
    end
end

type TagBatch; sgen; state; sent; word; done;
    function TagBatch(sgen, batchsize, ftype, dense)
        word = (dense ? zeros : sponehot)(ftype, maxtoken(sgen), batchsize)
        sent = Array(Any, batchsize)
        new(sgen, nothing, sent, word, false)
    end
end

import Base: start, done, next

# the TagData state is just nword, which is the number of words
# completed in the current batch.

function start(d::TagData)
    d.x.done = d.y.done = false
    d.x.state = start(d.x.sgen)
    d.y.state = start(d.y.sgen)
    nextbatch(d.x)
    nextbatch(d.y)
    return 0
end

# We stop when there is not enough data to fill a batch

done(d::TagData, state)=(d.x.done && d.y.done)

# TagData.next returns the next token

function next(d::TagData, nword)
    nword += 1
    maxlen = maximum(map(length, d.x.sent))
    if nword > maxlen
        nextbatch(d.x)
        nextbatch(d.y)
        return (nothing, 0)
    end
    for s=1:length(d.x.sent)
        n=length(d.x.sent[s])
        xword = (nword <= n ? d.x.sent[s][nword] : 0)
        yword = (nword <= n ? d.y.sent[s][nword] : 0)
        # @show s,n,xword,yword
        setrow!(d.x.word, xword, s)
        setrow!(d.y.word, yword, s)
        d.mask[s] = (xword == 0 ? 0 : 1)
    end
    return ((d.x.word, d.y.word, d.mask), nword)
end

function nextbatch(b::TagBatch)
    for i=1:length(b.sent)
        done(b.sgen, b.state) && (b.done=true; return)
        (b.sent[i], b.state) = next(b.sgen, b.state)
    end
end

maxtoken(s::TagData,i)=maxtoken(i==1 ? s.x.sgen : i==2 ? s.y.sgen : error())
maxtoken(x::Vector{Vector{Int}}) = maximum(map(maximum,x))

sponehot(ftype, m, n)=sparse(ones(Int32,n), Int32[1:n;], one(ftype), m, n)

# # TODO: these assume one hot columns, make them more general.
# setrow!(x::SparseMatrixCSC,i,j)=(i>0 ? (x.rowval[j] = i; x.nzval[j] = 1) : (x.rowval[j]=1; x.nzval[j]=0))
# setrow!(x::Array,i,j)=(x[:,j]=0; i>0 && (x[i,j]=1))

# function readvocab(file) # TODO: test with cmd e.g. `zcat foo.gz`
#     d = Dict{Any,Int}() 
#     open(file) do f
#         for l in eachline(f)
#             for w in split(l)
#                 get!(d, w, 1+length(d))
#             end
#         end
#     end
#     return d
# end
