"""
S2S(net::Function) implements the sequence to sequence model from:
Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence
learning with neural networks. Advances in neural information
processing systems, 3104-3112.  The @knet function `net` is duplicated
to create an encoder and a decoder.  See `S2SData` for data format.
"""
immutable S2S <: Model; encoder; decoder; params;
    function S2S(kfun::Function; o...)
        enc = Net(encoder; o..., f=kfun)
        dec = Net(decoder; o..., f=kfun)
        new(enc, dec, vcat(params(enc), params(dec)))
    end
end

params(m::S2S)=m.params
reset!(m::S2S; o...)=(reset!(m.encoder; o...);reset!(m.decoder; o...))

@knet function encoder(word; f=:lstm, hidden=0, o...)
    wvec = wdot(word; o..., out=hidden)
    hvec = f(wvec; o..., out=hidden)
end

@knet function decoder(word; f=:lstm, hidden=0, vocab=0, o...)
    hvec = encoder(word; o..., f=f, hidden=hidden)
    tvec = wdot(hvec; out=vocab)
    pvec = soft(tvec)
end

train(m::S2S, data, loss; o...)=s2s_loop(m, data, loss; trn=true, ystack=Any[], o...)

test(m::S2S, data, loss; o...)=(l=zeros(2); s2s_loop(m, data, loss; losscnt=l, o...); l[1]/l[2])

# todo: implement load/save: revive cpucopy - done. 
# todo: implement predict
# data should be a sequence generator for the source side
# i.e. next(data) should be a vector of source side ints
# todo: introduce an abstract type for data
# done: we need to have our own batch matrices
# done: the batchsize for the model needs to change
# note: no need for mask if we use batchsize=1 for predict
# note: s2sdata should probably be part of train, since we'll have to write different train functions for different models anyway, how the model consumes data is not part of the data, it is part of train, will make it easier to write data generators.
# done: copyforw needs to do repmat
# done: eos(sgen) different for x2
# todo: nbest should be another parameter, for now we assume nbest=1

function predict(m::S2S, sgen; beamsize=1, dense=false, nbest=1, o...)
    clear!(m)
    p = params(m)
    ftype = eltype(p[1].out)
    eos1 = vocab1 = size(p[1].out, 2)
    eos2 = vocab2 = size(p[end].out, 1)
    # @assert eos1 == eos(sgen)
    if dense
        x1 = zeros(ftype, vocab1, 1)
        x2 = zeros(ftype, vocab2, beamsize)
    else
        x1 = sponehot(ftype, vocab1, 1)
        x2 = sponehot(ftype, vocab2, beamsize)
    end
    pred = Any[]
    for seq in sgen
        reset!(m)
        forw(m.encoder, setrow!(x1,eos1,1); trn=false, seq=true, o...)
        for x in reverse(seq)
            forw(m.encoder, setrow!(x1, x, 1); trn=false, seq=true, o...)
        end
        for i in 1:beamsize
            setrow!(x2, eos2, i)
        end
        #@show display(to_host(m.encoder.out[end]))
        initforw(m.decoder, x2)
        s2s_copyforw!(m.encoder, m.decoder, [1 for i=1:beamsize])
        push!(pred, decode(m.decoder, x2; nbest=nbest, o...))
    end
    return pred
end

# done: construct the new h, need a shuffling version of copyforw, need to be careful not to overwrite
# done: construct the new x, need to find the top scoring words without repetition, these are cumulative scores, not word scores! and cumulative words, e.g. identical words may win continuing different sequences.  we can use bad initial scores for all but the first of the initial columns.
# todo: generate the max score sequence (already have prob in y) no need to log all: we do need to log all to combine with previous logp
# done: n-best is a different parameter than beamsize, implement n-best?
# done: pop the finished sequences from the beam, stop when everything on the beam is worse than top-n in the popped sequences.
# todo: give back the answer in string
# todo: add dropout

function decode(r::Net, x; nbest=1, o...)
    (nrows,ncols) = size(x)
    eos = vocab = nrows
    # TODO: some of these should not need to be allocated every time
    global ylogp = CudaArray(eltype(x), size(x))
    global yscore = Array(eltype(x), size(x))
    # beam is going to hold the previous tokens leading up to the current columns
    global beam = [ Int[] for i=1:ncols ]
    # score is the cumulative logp of sequences on the beam, we initialize all but the first column with -Inf because initially all columns are going to be the same.
    global score = Array(eltype(x), (1, ncols))
    fill!(score, -Inf); score[1]=0
    # nbest list for completed sequences, keys: Int[], values: cumulative logp.  When scores on the beam are all worse than the ones in nbest stop.
    global nbestqueue = PriorityQueue()

    while true
        #@show x
        #@show display(to_host(r.out[end-3]))
        yprob = forw(r, x; trn=false, seq=true, o...)
        log!(yprob, ylogp)
        copysync!(yscore, 1, ylogp, 1, length(yscore))
        broadcast!(+, yscore, score, yscore)
        global ytop = topn(yscore, 2*ncols)
        global newbeam,newscore,oldcol,newn
        newbeam,newscore,oldcol,newn = similar(beam),similar(score),zeros(Int,ncols),0
        global tmp = similar(x)        # todo: avoid alloc
        for i in ytop
            (iword, ibeam) = ind2sub(yscore, i)
            # println("yscore$((beam[ibeam]..., iword))=$(yscore[i])")
            if iword == eos
                if length(nbestqueue) < nbest
                    nbestqueue[beam[ibeam]] = yscore[i]
                elseif yscore[i] > peek(nbestqueue).second
                    dequeue!(nbestqueue)
                    nbestqueue[beam[ibeam]] = yscore[i]
                end
            else
                newn += 1
                oldcol[newn] = ibeam
                newbeam[newn] = vcat(beam[ibeam],iword)
                newscore[newn] = yscore[i]
                setrow!(tmp, iword, newn)
                newn == ncols && break
            end
        end
        # todo: what if there is not enough to fill ncols?
        @assert newn == ncols
        beam,score = newbeam,newscore
        length(nbestqueue) >= nbest && score[1] < peek(nbestqueue).second && break
        s2s_copyforw!(r, r, oldcol)
        x = tmp
    end
    @assert length(nbestqueue) == nbest
    global nbestlist = Array(Vector{Int}, nbest)
    for i=nbest:-1:1
        nbestlist[i] = dequeue!(nbestqueue)
    end
    return nbestlist
end

using Base.Collections

function topn{T}(a::Array{T},n::Int)
    p = PriorityQueue{Int,T,Base.Order.ForwardOrdering}()
    plen::Int = 0
    pmin::T = typemax(T)
    @inbounds for i=1:length(a)
        if plen < n
            p[i] = a[i]
            plen += 1
            pmin = peek(p).second
        elseif a[i] > pmin
            dequeue!(p)
            p[i] = a[i]
            pmin = peek(p).second
        end
    end
    k = Array(Int,n)
    @inbounds for i=n:-1:1
        k[i] = dequeue!(p)
    end
    return k
end

function clear!(m::S2S)
    for net in (m.encoder, m.decoder)
        for i in 1:nops(net)
            if !isa(net.op[i],Par) && !isa(net.op[i],Arr)
                net.out[i] = net.out0[i] = net.dif[i] = net.dif0[i] = net.tmp[i] = nothing
            end
        end
    end
    return m
end

function s2s_loop(m::S2S, data, loss; gcheck=false, o...)
    s2s_lossreport()
    decoding = false
    reset!(m; o...)
    for (x,ygold,mask) in data
        if decoding && ygold == nothing # the next sentence started
            gcheck && break
            s2s_eos(m, data, loss; gcheck=gcheck, o...)
            reset!(m; o...)
            decoding = false
        end
        if !decoding && ygold != nothing # source ended, target sequence started
            s2s_copyforw!(m)
            decoding = true
        end
        if decoding && ygold != nothing # keep decoding target
            s2s_decode(m, x, ygold, mask, loss; o...)
        end
        if !decoding && ygold == nothing # keep encoding source
            s2s_encode(m, x; o...)
        end
    end
    s2s_eos(m, data, loss; gcheck=gcheck, o...)
end

function s2s_encode(m::S2S, x; trn=false, o...)
    forw(m.encoder, x; trn=trn, seq=true, o...)
end    

function s2s_decode(m::S2S, x, ygold, mask, loss; trn=false, ystack=nothing, losscnt=nothing, o...)
    ypred = forw(m.decoder, x; trn=trn, seq=true, o...)
    ystack != nothing  && push!(ystack, (copy(ygold),copy(mask))) # TODO: get rid of alloc
    losscnt != nothing && s2s_loss(m, ypred, ygold, mask, loss; losscnt=losscnt, o...)
end

function s2s_loss(m::S2S, ypred, ygold, mask, loss; losscnt=nothing, lossreport=0, o...)
    (yrows, ycols) = size2(ygold)
    nwords = (mask == nothing ? ycols : sum(mask))
    losscnt[1] += loss(ypred,ygold;mask=mask) # loss divides total loss by minibatch size ycols.  at the end the total loss will be equal to
    losscnt[2] += nwords/ycols                # losscnt[1]*ycols.  losscnt[1]/losscnt[2] will equal totalloss/totalwords.
    # we could scale losscnt with ycols so losscnt[1] is total loss and losscnt[2] is total words, but I think that breaks gradcheck since the scaled versions are what gets used for parameter updates in order to prevent batch size from effecting step size.
    lossreport > 0 && s2s_lossreport(losscnt,ycols,lossreport)
end

s2s_time0 = s2s_time1 = s2s_inst = 0
s2s_print(a...)=(for x in a; @printf("%.2f ",x); end; println(); flush(STDOUT))

function s2s_lossreport()
    global s2s_time0, s2s_time1, s2s_inst
    s2s_inst = 0
    s2s_time0 = s2s_time1 = time_ns()
    println("time inst speed perp")
end

function s2s_lossreport(losscnt,batchsize,lossreport)
    global s2s_time0, s2s_time1, s2s_inst
    s2s_time0 == 0 && s2s_lossreport()
    losscnt == nothing && return
    losscnt[2]*batchsize < lossreport && return
    curr_time = time_ns()
    batch_time = Int(curr_time - s2s_time1)/10^9
    total_time = Int(curr_time - s2s_time0)/10^9
    s2s_time1 = curr_time
    batch_inst = losscnt[2]*batchsize
    batch_loss = losscnt[1]*batchsize
    s2s_inst += batch_inst
    s2s_print(total_time, s2s_inst, batch_inst/batch_time, exp(batch_loss/batch_inst))
    losscnt[1] = losscnt[2] = 0
end

function s2s_eos(m::S2S, data, loss; trn=false, gcheck=false, ystack=nothing, maxnorm=nothing, gclip=0, o...)
    if trn
        s2s_bptt(m, ystack, loss; o...)
        g = (gclip > 0 || maxnorm!=nothing ? gnorm(m) : 0)
        if !gcheck
            gclip=(g > gclip > 0 ? gclip/g : 0)
            update!(m; gclip=gclip, o...)
        end
    end
    if maxnorm != nothing
        w=wnorm(m)
        w > maxnorm[1] && (maxnorm[1]=w)
        g > maxnorm[2] && (maxnorm[2]=g)
    end
end

function s2s_bptt(m::S2S, ystack, loss; o...)
    while !isempty(ystack)
        (ygold,mask) = pop!(ystack)
        back(m.decoder, ygold, loss; seq=true, mask=mask, o...)
    end
    @assert m.decoder.sp == 0
    s2s_copyback!(m)
    while m.encoder.sp > 0
        back(m.encoder; seq=true, o...)
    end
end

function s2s_copyforw!(m::S2S)
    for n=1:nops(m.encoder)
        if forwref(m.encoder, n)
            # m.decoder.out0[n] == nothing && (m.decoder.out0[n] = similar(m.encoder.out[n]))
            # m.decoder.out[n] = copysync!(m.decoder.out0[n], m.encoder.out[n])
            m.decoder.out[n] = m.decoder.out0[n] = m.encoder.out[n]
        end
    end
end

function s2s_copyforw!(n1::Net, n2::Net, cols)
    ncols = length(cols)
    for n=1:nops(n1)
        if forwref(n1, n)
            o1 = n1.out[n]
            @assert o1 != nothing
            nrows = size(o1,1)
            o2 = n2.tmp[n]
            if o2 == nothing || size(o2) != (nrows, ncols)
                o2 = similar(o1, (nrows, ncols))
            end
            for i=1:length(cols)
                copysync!(o2, 1+(i-1)*nrows, o1, 1+(cols[i]-1)*nrows, nrows)
            end
            n2.tmp[n] = n2.out0[n]
            n2.out[n] = n2.out0[n] = o2
        end
    end
end

function s2s_copyback!(m::S2S)
    for n=1:nops(m.encoder)
        if forwref(m.encoder, n)
            # m.encoder.dif0[n] == nothing && (m.encoder.dif0[n] = similar(m.decoder.dif[n]))
            # m.encoder.dif[n] = copysync!(m.encoder.dif0[n], m.decoder.dif[n])
            m.encoder.dif[n] = m.encoder.dif0[n] = m.decoder.dif[n]
        end
    end
end

function Base.isequal(a::S2S,b::S2S)
    typeof(a)==typeof(b) || return false
    for n in fieldnames(a)
        if isdefined(a,n) && isdefined(b,n)
            isequal(a.(n), b.(n)) || return false
        elseif isdefined(a,n) || isdefined(b,n)
            return false
        end
    end
    return true
end

# FAQ:
#
# Q: How do we handle the transition?  Is eos fed to the encoder or the decoder?
#   It looks like the decoder from the picture.  We handle switch using the state variable.
# Q: Do we feed one best, distribution, or gold for word[t-1] to the decoder?
#   Gold during training, one-best output during testing?  (could also try the actual softmax output)
# Q: How do we handle the sequence ending?  How is the training signal passed back?
#   gold outputs, bptt first through decoder then encoder, triggered by output eos (or data=nothing when state=decode)
# Q: input reversal?
#   handled by the data generator
# Q: batching and padding of inputs of different length?
#   handled by the data generator
# Q: data format? <eos> ran dog The <eos> el perror corrio <eos> sentence next The <eos> its spanish version <eos> ...
#   so data does not need to have x/y pairs, just a y sequence for training (and testing?).
#   we could use nothings to signal eos, but we do need an actual encoding for <eos>
#   so don't replace <eos>, keep it, just insert nothings at the end of each sentence.
#   except in the very beginning, a state variable keeps track of encoding vs decoding
# Q: how about the data generator gives us x/y pairs:
#   during encoding we get x/nothing as desired input/output.
#   during decoding we get x[t-1]/x[t] as desired input/output.
#   this is more in line with what we did with rnnlm,
#   also in line with models not caring about how data is formatted, only desired input/output.
#   in this design we can distinguish encoding/decoding by looking at output
# Q: test time: 
#   encode proceeds normally
#   when we reach nothing, we switch to decoding
#   when we reach nothing again, we perform bptt (need stack for gold output, similar to rnnlm)
#   the first token to decoder gets fed from the gold (presumably <eos>)
#   the subsequent input tokens come from decoder's own output
#   second nothing resets and switches to encoding, no bptt
# Q: beam search?
#   don't need this for training or test perplexity
#   don't need this for one best greedy output
#   it will make minibatching more difficult
#   this is a problem for predict (TODO)
# Q: gradient scaling?
#   this becomes a problem once we start doing minibatching and padding short sequences.  I started
#   dividing loss and its gradients with minibatch size so the choice of minibatch does not effect
#   the step sizes and require a different learning rate.  In s2s if we have a sequence minibatch of
#   size N, which consists of T token minibatches each of size N.  Some sequences in the minibatch
#   are shorter so some of the token minibatches contain padding.  Say a particular token minibatch
#   contains n<N real words, the rest is padding.  Do we normalize using n or N?  It turns out in
#   this case the correct answer is N because we do not want the word in this minibatch to have
#   higher weight compared to other words in the sequence!
# Q: do we zero out padded columns?
#   We should set padding columns to zero in the input.  Otherwise when
#   the next time step has a legitimate word in that column, the hidden
#   state column will not be zero.


### DEAD CODE:
    # if losscnt != nothing 
    #     l = loss(ypred, ygold)  # returns total loss / num columns ignoring padding
    #     ycols = size(ygold,2)
    #     nzcols = 0
    #     for j=1:ycols; zerocolumn(ygold,j) || (nzcols += 1); end
    #     losscnt[1] += l # this has been scaled by (1/ycols) by the loss function
    #     losscnt[2] += (nzcols/ycols)    # this should work as long as loss for padded columns is 0 as it should be
    # end
