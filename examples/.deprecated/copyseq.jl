using Knet
module CopySeq
using Main, Knet, ArgParse
using Knet: copysync!
@useifgpu CUDArt
@useifgpu CUSPARSE

function main(args=ARGS)
    info("Learning to copy sequences to test the S2S model.")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("datafiles"; nargs='+'; required=true; help="First file used for training")
        ("--dictfile"; help="Dictionary file, first datafile used if not specified")
        ("--epochs"; arg_type=Int; default=1)
        ("--hidden"; arg_type=Int; default=100)
        ("--batchsize"; arg_type=Int; default=128)
        ("--lossreport"; arg_type=Int; default=0)
        ("--gclip"; arg_type=Float64; default=10.0)
        ("--lr"; arg_type=Float64; default=2.0)
        ("--fbias"; arg_type=Float64; default=0.0)
        ("--ftype"; default="Float32")
        ("--winit"; default="Gaussian(0,0.01)")
        ("--dense"; action=:store_true)
        ("--fast"; help="skip norm and loss calculations."; action=:store_true)
        ("--gcheck"; arg_type=Int; default=0)
        ("--seed"; arg_type=Int; default=42)
        ("--nosharing"; action = :store_true)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args, s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)
    dict = (dictfile == nothing ? datafiles[1] : dictfile)
    global data = Any[]
    for f in datafiles
        push!(data, S2SData(f; batchsize=batchsize, ftype=eval(parse(ftype)), dense=dense, dict=dict))
    end
    vocab = maxtoken(data[1],2)
    # global model = S2S(lstm; fbias=fbias, hidden=hidden, vocab=vocab, winit=eval(parse(winit)))
    global model = compile(:copyseq; fbias=fbias, out=hidden, vocab=vocab, winit=eval(parse(winit)))
    setp(model; lr=lr)
    if nosharing
        set!(model, :forwoverwrite, false)
        set!(model, :backoverwrite, false)
    end

    perp = zeros(length(data))
    (maxnorm,losscnt) = fast ? (nothing,nothing) : (zeros(2),zeros(2))
    t0 = time_ns()
    println("epoch  secs    ptrain  ptest.. wnorm  gnorm")
    myprint(a...)=(for x in a; @printf("%-6g ",x); end; println(); flush(STDOUT))
    for epoch=1:epochs
        fast || (fill!(maxnorm,0); fill!(losscnt,0))
        train(model, data[1], softloss; gclip=gclip, maxnorm=maxnorm, losscnt=losscnt, lossreport=lossreport)
        fast || (perp[1] = exp(losscnt[1]/losscnt[2]))
        for d=2:length(data)
            loss = test(model, data[d], softloss)
            perp[d] = exp(loss)
        end
        myprint(epoch, (time_ns()-t0)/1e9, perp..., (fast ? [] : maxnorm)...)
        gcheck > 0 && gradcheck(model,
                                f->(train(f,data[1],softloss;losscnt=fill!(losscnt,0),gcheck=true);losscnt[1]),
                                f->(test(f,data[1],softloss;losscnt=fill!(losscnt,0),gcheck=true);losscnt[1]);
                                gcheck=gcheck)
    end
    return (fast ? (perp...) :  (perp..., maxnorm...))
end

# This copies lstm exactly for replicatability:
@knet function copyseq(word; fbias=0, vocab=0, o...)
    if !decoding
        x = wdot(word; o...)
        input  = wbf2(x,h; o..., f=:sigm)
        forget = wbf2(x,h; o..., f=:sigm, binit=Constant(fbias))
        output = wbf2(x,h; o..., f=:sigm)
        newmem = wbf2(x,h; o..., f=:tanh)
    else
        x = wdot(word; o...)
        input  = wbf2(x,h; o..., f=:sigm)
        forget = wbf2(x,h; o..., f=:sigm, binit=Constant(fbias))
        output = wbf2(x,h; o..., f=:sigm)
        newmem = wbf2(x,h; o..., f=:tanh)
    end
    cell = input .* newmem + cell .* forget
    h  = tanh(cell) .* output
    if decoding
        tvec = wdot(h; out=vocab)
        return soft(tvec)
    end
end

@knet function copyseq1(word; fbias=0, vocab=0, o...)
    if decoding
        x = wdot(word; o...)
        input  = sigm(aff2(x,h; o...))
        forget = sigm(aff2(x,h; o..., binit=Constant(fbias)))
        output = sigm(aff2(x,h; o...))
        newmem = tanh(aff2(x,h; o...))
    else
        x = wdot(word; o...)
        input  = sigm(aff2(x,h; o...))
        forget = sigm(aff2(x,h; o..., binit=Constant(fbias)))
        output = sigm(aff2(x,h; o...))
        newmem = tanh(aff2(x,h; o...))
    end
    c = input .* newmem + c .* forget
    h  = tanh(c) .* output
    if decoding
        return soft(wdot(h; out=vocab))
    end
end

@knet function aff2(x,y; out=0, o...)
    a = par(; o..., dims=(out,0))
    b = par(; o..., dims=(out,0))
    c = par(; o..., dims=(0,))
    # return a*x+b*y+c
    xy = a*x+b*y
    return c+xy
end

function train(m, data, loss; o...)
    s2s_loop(m, data, loss; trn=true, ystack=Any[], o...)
end

function test(m, data, loss; losscnt=zeros(2), o...)
    s2s_loop(m, data, loss; losscnt=losscnt, o...)
    losscnt[1]/losscnt[2]
end

# Persistent storage for ygold and mask
s2s_ygold = nothing
s2s_mask = nothing
@gpu copytogpu(y,x::Array)=CudaArray(x)
@gpu copytogpu(y,x::SparseMatrixCSC)=CudaSparseMatrixCSC(x)
@gpu copytogpu{T}(y::CudaArray{T},x::Array{T})=(size(x)==size(y) ? copysync!(y,x) : copytogpu(nothing,x))
@gpu copytogpu{T}(y::CudaSparseMatrixCSC{T},x::SparseMatrixCSC{T})=(size(x)==size(y) ? copysync!(y,x) : copytogpu(nothing,x))


function s2s_loop(m, data, loss; gcheck=false, o...)
    global s2s_ygold, s2s_mask
    s2s_lossreport()
    decoding = false
    reset!(m)
    for (x,ygold,mask) in data
        nwords = (mask == nothing ? size(x,2) : sum(mask))
        # x,ygold,mask are cpu arrays; x gets copied to gpu by forw; we should do the other two here
        if ygold != nothing && gpu()
            ygold = s2s_ygold = copytogpu(s2s_ygold,ygold)
            mask != nothing && (mask = s2s_mask  = copytogpu(s2s_mask,mask)) # mask not used when ygold=nothing
        end
        if decoding && ygold == nothing # the next sentence started
            gcheck && break
            s2s_eos(m, data, loss; gcheck=gcheck, o...)
            reset!(m)
            decoding = false
        end
        if !decoding && ygold != nothing # source ended, target sequence started
            # s2s_copyforw!(m)
            decoding = true
        end
        if decoding && ygold != nothing # keep decoding target
            s2s_decode(m, x, ygold, mask, nwords, loss; o...)
        end
        if !decoding && ygold == nothing # keep encoding source
            s2s_encode(m, x; o...)
        end
    end
    s2s_eos(m, data, loss; gcheck=gcheck, o...)
end

function s2s_encode(m, x; trn=false, o...)
    # forw(m.encoder, x; trn=trn, seq=true, o...)
    (trn?sforw:forw)(m, x; decoding=false)
end    

function s2s_decode(m, x, ygold, mask, nwords, loss; trn=false, ystack=nothing, losscnt=nothing, o...)
    # ypred = forw(m.decoder, x; trn=trn, seq=true, o...)
    ypred = (trn?sforw:forw)(m, x; decoding=true)
    ystack != nothing  && push!(ystack, (copy(ygold),copy(mask))) # TODO: get rid of alloc
    losscnt != nothing && s2s_loss(m, ypred, ygold, mask, nwords, loss; losscnt=losscnt, o...)
end

function s2s_loss(m, ypred, ygold, mask, nwords, loss; losscnt=nothing, lossreport=0, o...)
    (yrows, ycols) = size2(ygold)  # TODO: loss should handle mask, currently only softloss does.
    losscnt[1] += loss(ypred,ygold;mask=mask) # loss divides total loss by minibatch size ycols.  at the end the total loss will be equal to
    losscnt[2] += nwords/ycols                # losscnt[1]*ycols.  losscnt[1]/losscnt[2] will equal totalloss/totalwords.
    # we could scale losscnt with ycols so losscnt[1] is total loss and losscnt[2] is total words, but I think that breaks gradcheck since the scaled versions are what gets used for parameter updates in order to prevent batch size from effecting step size.
    lossreport > 0 && s2s_lossreport(losscnt,ycols,lossreport)
end

function s2s_eos(m, data, loss; trn=false, gcheck=false, ystack=nothing, maxnorm=nothing, gclip=0, o...)
    if trn
        s2s_bptt(m, ystack, loss; o...)
        g = (gclip > 0 || maxnorm!=nothing ? gnorm(m) : 0)
        if !gcheck
            gscale=(g > gclip > 0 ? gclip/g : 1)
            update!(m; gscale=gscale, o...)
        end
    end
    if maxnorm != nothing
        w=wnorm(m)
        w > maxnorm[1] && (maxnorm[1]=w)
        g > maxnorm[2] && (maxnorm[2]=g)
    end
end

function s2s_bptt(m, ystack, loss; o...)
    while !isempty(ystack)
        (ygold,mask) = pop!(ystack)
        # back(m.decoder, ygold, loss; seq=true, mask=mask, o...)
        sback(m, ygold, loss; mask=mask, o...) # back passes mask on to loss
    end
    # @assert m.decoder.sp == 0
    # s2s_copyback!(m)
    # while m.encoder.sp > 0
    while m.sp > 0
        # back(m.encoder; seq=true, o...)
        sback(m)                # TODO: what about mask here?
        # error(:ok)
    end
end

s2s_time0 = s2s_time1 = s2s_inst = 0

function s2s_lossreport()
    global s2s_time0, s2s_time1, s2s_inst
    s2s_inst = 0
    s2s_time0 = s2s_time1 = time_ns()
    # println("time inst speed perp")
end

s2s_print(a...)=(for x in a; @printf("%.2f ",x); end; println(); flush(STDOUT))

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


!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)

end # module

### DEAD CODE

    # info("Warm-up epoch")
    # f=datafiles[1]; mini = S2SData(f, f; batch=batchsize, ftype=eval(parse(ftype)), dense=dense, dict1=dict[1], dict2=dict[2], stop=3200)
    # @date train(model, mini, softloss; gcheck=gcheck, gclip=gclip, getnorm=getnorm, getloss=getloss) # pretrain to compile for timing
    # info("Starting profile")
