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

@knet function encoder(word; f=nothing, hidden=0, o...)
    wvec = wdot(word; o..., out=hidden)
    hvec = f(wvec; o..., out=hidden)
end

@knet function decoder(word; f=nothing, hidden=0, vocab=0, o...)
    hvec = encoder(word; o..., f=f, hidden=hidden)
    tvec = wdot(hvec; out=vocab)
    pvec = soft(tvec)
end

train(m::S2S, data, loss; o...)=s2s_loop(m, data, loss; trn=true, ystack=Any[], o...)

test(m::S2S, data, loss; o...)=(l=zeros(2); s2s_loop(m, data, loss; losscnt=l, o...); l[1]/l[2])

function s2s_loop(m::S2S, data, loss; gcheck=false, o...)
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
    if lossreport > 0 && losscnt[2]*ycols > lossreport
        println((exp(losscnt[1]/losscnt[2]), losscnt[1]*ycols, losscnt[2]*ycols))
        losscnt[1] = losscnt[2] = 0
    end
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
            # m.decoder.out[n] = copy!(m.decoder.out0[n], m.encoder.out[n])
            m.decoder.out[n] = m.decoder.out0[n] = m.encoder.out[n]
        end
    end
end

function s2s_copyback!(m::S2S)
    for n=1:nops(m.encoder)
        if forwref(m.encoder, n)
            # m.encoder.dif0[n] == nothing && (m.encoder.dif0[n] = similar(m.decoder.dif[n]))
            # m.encoder.dif[n] = copy!(m.encoder.dif0[n], m.decoder.dif[n])
            m.encoder.dif[n] = m.encoder.dif0[n] = m.decoder.dif[n]
        end
    end
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
