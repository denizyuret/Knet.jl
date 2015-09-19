# TODO
# - performance: figure out when no back needed, no returndx needed
# - testing: add a testnet.jl that constructs random nets and does gradient testing: testlayers could not find the pool bug
# - performance: do better register optimization
# - if an op is using an external array, it should not store it.
# - if we take xi/yi as a parameter for back maybe the net would not have to remember it?
# - rename train->trn for ops

type Net <: Model; op; inputs; ninputs; params; push; multi; out; out0; dif; dif0; dif1; stack; sp; dbg; Net()=new(); end

### Net functions: params, ninputs, nops, op, loss, forw, back

params(r::Net)=r.params
ninputs(r::Net)=r.ninputs
nops(r::Net)=length(r.op)
op(r::Net,n)=r.op[n]

# forw(r::Net,x::Vector) for sequence.
# x can be a Vector of Arrays representing items.
# x can be a Vector of Tuples representing multiple inputs.
# x cannot be a Vector of scalars (TODO:think this over)

function forw(r::Net, x::Vector; yout=nothing, ygold=nothing, a...)
    # display((:forwseq0,length(x),vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
    isbits(eltype(x)) && error("forw expects a minibatch")
    x1 = (isa(x[1], Tuple) ? x[1] : (x[1],))
    initforw(r, x1...; a...)
    loss = 0.0
    for i=1:length(x)
        xi = (isa(x[i], Tuple) ? x[i] : (x[i],))
        yi = (yout == nothing ? nothing : yout[i])
        yg = (ygold == nothing ? nothing : ygold[i])
        loss += forw(r, xi...; seq=true, yout=yi, ygold=yg, a...)
    end
    # display((:forwseq1,length(x),vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
    return loss
end

# forw(r::Net,x...) for individual items that may or may not be part
# of a sequence.

function forw(r::Net, inputs...; yout=nothing, ygold=nothing, seq=false, trn=false, a...)
    # display((:forw0,seq,vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
    length(inputs) == ninputs(r) || error("Wrong number of inputs")
    seq || initforw(r, inputs...; a...)
    N = nops(r)
    for i = 1:ninputs(r)
        n = i+N                           # input[i] goes into out[i+N]
        eltype(inputs[i]) == eltype(r.out0[n]) || error("Element type mismatch $i $n")
        trn && r.push[n] && push(r,n)         # t:140 save old input if necessary
        r.out[n] = copy!(r.out0[n], inputs[i]) 	# ; dbg(r,:out,n) # t:98 inputs can be any type of array, this will copy it to gpu or wherever
    end
    for n = 1:N
        trn && r.push[n] && push(r,n)         # t:327
        # TODO: forw.op interface differences: returning y, train/trn, y/yout, ...
        r.out[n] = forw(r.op[n], r.out[r.inputs[n]]...; y=r.out0[n], train=trn, a...)     # ;dbg(r,:out,n) # t:2300
    end
    # display((:forw1,seq,vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
    yout != nothing && copy!(yout, r.out[N])
    ygold == nothing && return 0.0
    loss(r.op[N], ygold; y=r.out[N])
end

# initforw(r::Net,x...) is called at the beginning of a sequence or
# before processing a stand-alone item.  It is not called between
# elements of a sequence.

function initforw(r::Net, inputs...; keepstate=false, a...)
    # display((:initforw0,keepstate,vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
    r.sp == 0 || error("Stack corruption")
    # We allocate and/or resize r.out0
    out = fill!(cell(length(r.out0)), nothing)          # TODO: can we get rid of alloc
    for i = 1:ninputs(r)
        n = i+nops(r)
        out[n] = initarray(r.out0, n, inputs[i])
    end
    while findfirst(out,nothing) > 0
        nalloc = 0
        for n = 1:nops(r)
            out[n] == nothing || continue
            s = ysize(r.op[n], out[r.inputs[n]]...)
            s == nothing && continue        # may happen with recurrent connections
            out[n] = initarray(r.out0, n, out[r.inputs[n]][1], s; dense=true)
            nalloc += 1
        end
        nalloc == 0 && error("Cannot determine size of array")
    end
    # We recover or reset r.out:
    keepstate ? copy!(r.out, r.out0) : fill!(r.out, nothing)
    # display((:initforw1,keepstate,vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
end

# back(r::Net,dy::Vector) for a sequence
# TODO: truncated bptt
# - go forward k1 steps, run back for k2, update, recover state
# - if k1==k2 we just need the keepstate option to forw
# - if k1>k2 the stack won't be cleared
# - if k1<k2 the stack will be overdrawn

function back(r::Net, dy::Vector; dx=nothing, a...)
    # display((:backseq0,length(dy),vecnorm0(r.dif),vecnorm0(r.stack[1:r.sp])))
    initback(r; seq=true, a...)
    for i=length(dy):-1:1
        dxi = (dx == nothing ? nothing : dx[i])
        back(r, dy[i]; seq=true, dx=dxi, a...)
    end
    # display((:backseq1,length(dy),vecnorm0(r.dif),vecnorm0(r.stack[1:r.sp])))
end

# back(r::Net,dy) for individual items that may or may not be elements
# of a sequence.

function back(r::Net, dy; dx=nothing, seq=false, a...)
    # display((:back0,seq,vecnorm0(r.dif),vecnorm0(r.stack[1:r.sp])))
    dx == nothing || length(dx) == ninputs(r) || error("Wrong number of inputs")
    seq || initback(r; seq=false, a...)
    N = nops(r)
    if dy == nothing
        r.multi[N] || (r.dif[N] = nothing)
    elseif eltype(dy) != eltype(r.dif0[N])
        error("Element type mismatch dy:$(eltype(dy)) dif0[$N]:$(eltype(r.dif0[N]))")
    elseif r.multi[N]
        copy!(r.dif1[N], dy)
        r.dif[N] = axpy!(1,r.dif1[N],r.dif0[N])
    else
        r.dif[N] = copy!(r.dif0[N], dy)
    end										; dbg(r,:dif,N) 
    for n = N:-1:1
        if r.dif[n] == nothing
            for i in r.inputs[n]
                r.multi[i] || (r.dif[i] = nothing)
            end
        else
            dxn = Any[]
            for i in r.inputs[n]
                push!(dxn, r.multi[i] ? r.dif1[i] : r.dif0[i])
            end
            back(r.op[n], r.dif[n]; incr=seq, x=get1(r.out[r.inputs[n]]), y=r.out[n], dx=get1(dxn), a...) # t:2164
            for i in r.inputs[n]
                r.multi[i] && axpy!(1, r.dif1[i], r.dif0[i])            ; r.multi[i]&&dbg(r,:dif1,i)
                r.dif[i] = r.dif0[i]                                    ; dbg(r,:dif,i)
            end
            r.multi[n] && fill!(r.dif[n],0)                           ; r.multi[n]&&dbg(r,:dif,n) # t:157
        end
        r.push[n] && pop(r,n)                                    ; r.push[n]&&dbg(r,:out,n)
    end
    for i = ninputs(r):-1:1
        n = i+N
        r.push[n] && pop(r,n)                                    ; r.push[n] && dbg(r,:out,n)
        dx == nothing || copy!(dx[i], r.dif[n])
    end
    # display((:back1,seq,vecnorm0(r.dif),vecnorm0(r.stack[1:r.sp])))
    return dx
end

# initback(r::Net) called at the beginning of a sequence or a
# stand-alone item, never between elements of a sequence.

function initback(r::Net; seq=false, a...)
    # display((:initback0,seq,vecnorm0(r.dif),vecnorm0(r.stack[1:r.sp])))
    fill!(r.dif, nothing)                           # why? (TODO)
    for n=1:length(r.dif0)
        initarray(r.dif0, n, r.out0[n])
        if r.multi[n]
            initarray(r.dif1, n, r.out0[n])
        end
    end
    for n=1:length(r.dif0)
        r.multi[n] && fill!(r.dif0[n], 0)           # zeroed by back at every item in sequence
    end
    for w in params(r)
        similar!(w, :diff, w.arr)
    end
    if seq
        for w in params(r)
            similar!(w, :inc, w.arr)
            fill!(w.diff, 0)                            # zeroed only once at the beginning of the sequence
        end
    end
    # display((:initback1,seq,vecnorm0(r.dif),vecnorm0(r.stack[1:r.sp])))
end


### Stack functions: push, pop

function push(r::Net,n::Int)
    length(r.stack) <  r.sp && error("Stack error")
    length(r.stack) == r.sp && push!(r.stack, :newcell)
    r.sp += 1
    if r.out[n] == nothing                             # TODO: (minor) remove these checks once code is tested
        r.stack[r.sp] == nothing || r.stack[r.sp] == :newcell || warn("pushing nothing over array")
        r.stack[r.sp] = nothing
    elseif r.stack[r.sp] == nothing
        warn("pushing array over nothing")
        r.stack[r.sp] = copy(r.out[n])
    elseif r.stack[r.sp] == :newcell
        r.stack[r.sp] = copy(r.out[n])
    elseif size(r.out[n]) != size(r.stack[r.sp])
        warn("pushing array of different size")
        resize!(r.stack[r.sp], size(r.out[n]))
        copy!(r.stack[r.sp], r.out[n])
    else
        copy!(r.stack[r.sp], r.out[n])
    end
end

function pop(r::Net,n::Int)
    r.sp > 0 || error("Stack error")
    if r.out[n] == nothing
        r.stack[r.sp] == nothing || warn("popping array over nothing")
    elseif r.stack[r.sp] == nothing
        # warn("popping nothing over array")
    elseif size(r.out[n]) != size(r.stack[r.sp])
        warn("popping different sized array")
    end
    r.out[n] = r.stack[r.sp]
    r.sp -= 1
end

### Net compiler

function Net(a...; o...)
    r = Net()
    initop(r, a...)
    initinputs(r, a...)
    @assert length(r.op)==length(r.inputs)
    initninputs(r)
    initparams(r)
    initpush(r)
    initmulti(r)
    initout(r)
    initdif(r)
    initstack(r)
    r.dbg = false
    setparam!(r; o...)
    return r
end

# r.op[n] is the n'th operation in the net
# The user specifies the operations in the MLP constructor arguments
# If an argument is another MLP, its operations are spliced in

function initop(r::Net, a...)
    r.op = Op[]
    for ai in a
        isa(ai,Tuple) && (ai=ai[1])
        isa(ai,Op) ?  push!(r.op, ai) :
        isa(ai,Net)   ?  append!(r.op, ai.op) :
        error("Bad op: $ai")
    end
end

# r.inputs[n] is an array of k indices for the inputs of the n'th op.
# k is typically 1 but can be 2 for Add2 and Mul2 operations
# index i<=nops(r) indicates output of r.op[i], i>nops(r) indicates network inputs
# By default an op takes the results of the previous k op outputs
# (or network inputs for initial ops)
# The user can override this by providing a tuple argument for an op
# The first element of the tuple is the op, the rest are user indices for inputs
# userindex j>0 indicates output of userarg[j], j<=0 indicates network input 1-j.

function initinputs(r::Net, a...)
    newindex = Array(Int, length(a))
    lastindex = 0
    for i=1:length(a)
        ai = isa(a[i],Tuple) ? a[i][1] : a[i]
        lastindex += (isa(ai,Op) ? 1 : length(ai.op))
        newindex[i] = lastindex
    end
    r.inputs = Any[]
    for i=1:length(a)
        ai = isa(a[i],Tuple) ? a[i][1] : a[i]
        bi = isa(a[i],Tuple) ? a[i][2:end] : ((i-ninputs(ai)):(i-1))
        length(bi) == ninputs(ai) || error("Wrong number of inputs for $i:$(typeof(ai))")
        if isa(ai, Op)
            push!(r.inputs, map(j->(j>0 ? newindex[j] : nops(r)+1-j), Int[bi...]))
        else
            j0 = length(r.inputs)
            for aii in ai.inputs
                push!(r.inputs, map(j->(j<=nops(ai) ? j+j0 : (j=bi[j-nops(ai)]; j>0 ? newindex[j] : nops(r)+1-j)), aii))
            end
        end
    end
end

# r.ninputs is the number of inputs the whole MLP expects
# indices i>nops(r) in r.inputs refer to network inputs

function initninputs(r::Net)
    n = 0
    for ai in r.inputs
        for aij in ai
            aij - nops(r) > n && (n = aij - nops(r))
        end
    end
    r.ninputs = n
end

# r.params points to all op parameters

function initparams(r::Net)
    r.params = Any[]
    for o in r.op
        append!(r.params, params(o))
    end
end

# r.push[n] is true if the result of op[n] (for n <= nops(r))
# or the network input n-nops(r) (for n > nops(r))
# should be saved for back calculation

function initpush(r::Net)
    r.push = falses(nops(r)+ninputs(r))
    for n=1:nops(r)
        back_reads_y(r.op[n]) && (r.push[n] = true)
        if back_reads_x(r.op[n])
            for i in r.inputs[n]
                r.push[i] = true
            end
        end
    end
end

# r.multi[n] is true if out[n] has fanout > 1.
# In which case its dif should be incrementally updated.

function initmulti(r::Net)
    nout = zeros(Int, nops(r)+ninputs(r))
    nout[nops(r)] = 1  # count network output
    for n=1:nops(r)
        for i in r.inputs[n]
            nout[i] += 1
        end
    end
    r.multi = (nout .> 1)
end

# r.out[n] is the array that holds the output of op[n] (for n<=nops(r))
# or the network input n-nops(r) (for n>nops(r))
# r.out0[n] holds the actual array, allowing r.out[n]==nothing when necessary
# we optimize memory use by sharing arrays when we can
# arrays are initialized empty at this point: 
# waiting for the first input to decide their final type and size

function initout(r::Net)
    nout = nops(r)+ninputs(r)
    index = zeros(Int,nops(r))          # index==0 represents need for new register
    for n=1:nops(r)
        r.push[n] && continue           # a saved register should only be written by op[n]
        if overwrites(r.op[n])
            i = r.inputs[n][1]          # see if we can overwrite the first input
            r.push[i] && continue       # do not overwrite if you are going to save for back
            ow = true
            k = n
            while true # see if anybody else uses i before its next update
                k = mod1(k+1, nops(r))
                in(i, r.inputs[k]) && (ow = false; break)
                k == i && break
                k == nops(r) && i > nops(r) && break
            end # fencepost check: N==1:OK, i==n:OK, i>N:OK
            ow && (index[n]=i)
        end
        # TODO: This is suboptimal, gives 13 regs for LSTM
        # Should look for existing regs no longer used
        # this is nontrivial because the sizes are unknown
    end

    r.out0 = cell(nout)
    for n=1:nops(r)
        index[n]==0 && (r.out0[n] = Any[])
    end
    for n=1:nops(r)                     # other ops will overwrite existing registers
        index[n]==0 && continue
        k = index[n]
        while index[k]!=0; k=index[k]; end
        r.out0[n] = r.out0[k]
    end
    for n=1:ninputs(r)                  # use the last registers for network inputs
        r.out0[n+nops(r)] = Any[]
    end
    r.out = fill!(cell(nout), nothing)
end

# r.dif[n] is the loss gradient of the last output of op[n] (for n<=nops(r))
# or the network input n-nops(r) (for n>nops(r))
# r.dif0[n] holds the actual array, r.dif[n] can point to this or be 'nothing' representing zero matrix
# r.dif1[n] is an extra array for incremental updates if r.multi[n]

function initdif(r::Net)
    # TODO: this is not optimal for add: no need for two copies of dy when neither input is overwriting.
    nout = nops(r)+ninputs(r)           # 1..nops(r) for op outputs, nops(r)+1..nops(r)+ninputs(r) for network inputs
    index = zeros(Int,nout)             # index==0 represents need for new register
    for n=1:nops(r)                     # find out if back(op[n]) can overwrite dif[n]
        overwrites(r.op[n]) || continue # overwrites means potentially both forw x<-y and back dy<-dx
        r.multi[n] && continue           # don't share dif[n] with multi-output
        for i in r.inputs[n]
            r.multi[i] && continue       # don't share dif[n] with multi-output
            index[i]==0 || error("should not happen")
            index[i]=n; break           # op[n] will overwrite dif[n] to get dif[i]
        end
    end
    r.dif0 = cell(nout)
    for n=1:nout
        index[n]==0 && (r.dif0[n] = Any[])
    end
    for n=1:nout
        if index[n] > 0
            k = index[n]
            while index[k]!=0; k=index[k]; end
            r.dif0[n] = r.dif0[k]
        end
    end
    r.dif1 = cell(nout)
    for n=1:length(r.dif1)
        r.dif1[n] = r.multi[n] ? Any[] : nothing
    end
    r.dif = fill!(cell(nout), nothing)
end

function initstack(r::Net)
    r.stack = Any[]
    r.sp = 0
end


### General utilities:

function initarray(a, i, x, dims=size(x); dense=false)
    if isempty(a[i])
        oldai = a[i]
        at = (gpu()?CudaArray:Array)
        xt = eltype(x)
        a[i] = (!dense && issparse(x) ? KUsparse(at,xt,dims) : fill!(KUdense(at, xt, dims),0))
        for j=1:length(a); a[j]===oldai && (a[j]=a[i]); end # preserve array sharing
    elseif eltype(a[i]) != eltype(x)
        error("Element type mismatch")
    elseif size(a[i]) != dims
        warn("Resizing $(size(a[i]))->$dims")
        fill!(resize!(a[i], dims), 0)
    end
    return a[i]
end

import Base: isequal

function isequal(a::Net, b::Net)
    for n in fieldnames(a)
        if isdefined(a,n) && isdefined(b,n)
            isequal(a.(n), b.(n)) || return false
        elseif isdefined(a,n) || isdefined(b,n)
            return false
        end
    end
    return true
end

# Do not copy inputs to gpu:

function gpucopy_internal(x::Net, stackdict::ObjectIdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    y = ccall(:jl_new_struct_uninit, Any, (Any,), Net)
    stackdict[x] = y
    for i in fieldnames(x)
        if isdefined(x,i)
            y.(i) = (i == :inputs ?
                     cpucopy_internal(x.(i), stackdict) :
                     gpucopy_internal(x.(i), stackdict))
        end
    end
    return y
end

get1(x)=(length(x)==1?x[1]:x)

### DEBUGGING

ptr16(x)=hex(x==nothing ? 0 : hash(pointer(x)) % 0xffff, 4)
ptr8(x)=hex(x==nothing ? 0 : hash(pointer(x)) % 0xff, 2)
idx1(x)=(x==nothing ? -1 : atype(x)==CudaArray ? to_host(x)[1] : atype(x)==Array ? x[1] : error("$(typeof(x))"))
vecnorm0(x)=map(xi->(xi==nothing ? 0 : floor(1e4*vecnorm(xi))/1e4),x)
# TODO: look into julia nullables to make this nothing=zero matrix thing better

function dbg(r,f,n)
    r.dbg || return
    a = r.(f)[n]
    print("\n==> $f[$n]($(ptr16(a))) stack:")
    println(map(ptr16, r.stack[1:r.sp]))
    a != nothing && display(convert(Array,a))
    if n <= nops(r) && !isempty(params(r.op[n]))
        p = params(r.op[n])[1]
        println("\nw[$n]($(ptr16(p.arr)))")
        display(convert(Array,p.arr))
        if isdefined(p,:diff)
            println("\ndw[$n]($(ptr16(p.diff)))")
            display(convert(Array,p.diff))
        end
    end
end

### DEAD CODE

# forw{T<:Number}(r::Net,  x::Vector{T}; a...)=error("forw expects a minibatch") # forw(r, reshape(x,  length(x),  1); a...)
# back{T<:Number}(r::Net, dy::Vector{T}; a...)=error("back expects a minibatch") # back(r, reshape(dy, length(dy), 1); a...)

# function back(r::Net, dy; o...)                 # dy[1:T] loss gradients for output y
#     (N,T) = (length(r.net), length(dy))
#     @assert length(r.x) == T
#     inithidden(r, :dh, N, 1)                    # r.dh[1:N] are loss gradients for hidden r.h[1:N,t]
#     initdiff(r)                                 # initialize weight gradients dw to zero
#     for t = T:-1:1
#         r.dh[N] = dy[t]                         # y[t]=r.h[N,t] so dy[t]=r.dh[N]
#         for n = N:-1:1
#             r.dh[n] == nothing && continue      # 'nothing' represents 0 loss gradient
#             ni = ninputs(r.net[n])
#             x = rnninput(r,n,t)
#             ni==1 && (x=x[1])                   # handle single input vs tuple input layers
#             dx = back(r.net[n], r.dh[n]; x=x, y=r.h[n,t], incr=true) # incr=true increments dw 
#             ni==1 && (dx=(dx,))                 # handle single input vs tuple input layers
#             for j=1:ni                          # set r.dh using dx for each input of n
#                 i=r.inputs[n][j]                # i is the j'th input of n
#                 if (r.dh[i] == nothing)         # dh[i] should be updated using dx[j]
#                     r.dh[i] = dx[j]
#                 elseif length(r.freads[i]) == 1
#                     r.dh[i] = dx[j]
#                 else
#                     axpy!(1, dx[j], r.dh[i])    # increment dh[i] if i has more than one output
#                 end
#             end
#             if length(r.freads[n]) > 1
#                 fill!(r.dh[n], 0)               # if n has multiple outputs, reset dh[n]
#             end
#         end
#     end
# end

# function initparams(r::Net, inputs...; seq=false)
#     for n = 1:nops(r)
#         p = params(r.op[n])
#         if !isempty(p) && findfirst(isempty,p)>0
#             forw(r.op[n], r.out0[r.inputs[n]]...; y=r.out0[n])
#         end
#     end
#     for w in params(r)
#         similar!(w, :diff, w.arr)
#         seq && similar!(w, :inc, w.arr)
#     end
# end

# # r.buf[n] is the array pre-allocated for register r.reg[n]
# # r.buf is initialized in three steps:
# # - initbuf determines the indices for array reuse.
# #   i=0 means allocate, i>0 means reuse r.buf[i]
# # - allocbuf allocates 0 size KUdense arrays
# #   this is a separate step because we need input x for type info
# # - the actual op resize the bufs as needed.

# # DONE: this should also look at back needs when allocating

# function initbuf(r::Net)
#     N = length(r.op)
#     index = fill!(Array(Any,N), 0)
#     for n=1:N
#         if overwrites(r.op[n])
#             index[n] = i = r.inputs[n][1] # tentatively prepare to overwrite first input
#             k = n  
#             while true # see if anybody else uses i before its next update
#                 k = mod1(k+1, N)
#                 in(i, r.inputs[k]) && (index[n] = 0; break)
#                 k == i && break
#             end # fencepost check: N==1:OK, i==n:OK
#         end
#     end
#     # info("$(sum(index.==0)) bufs will be allocated.")
#     return index
# end

# function allocbuf(r::Net, x)
#     isa(r.buf[1], Int) || return
#     N = length(r.op)
#     for n=1:N
#         if r.buf[n] == 0
#             r.buf[n] = KUdense(atype(x[1]), eltype(x[1]), (0,))
#         end
#     end
#     for n=1:N
#         isa(r.buf[n], Int) || continue
#         k = n
#         while isa(r.buf[k], Int)
#             k = r.buf[k]
#         end
#         r.buf[n] = r.buf[k]
#     end
# end

# # r.output[n] points to the last output of r.op[n]
# # its value could be an array (provided by r.buf[n]) or 'nothing'
# # 'nothing' is used to represent the zero matrix at time=0
# # and others that result from them at time>0

# function initoutput(r::Net)
#     r.output = fill!(Array(Any,length(r.op)), nothing)
# end

# # r.stack[n<=N][t] is a copy of the output of r.op[n] at r.time=t
# # r.stack[n>N][t] is a copy of the n-N'th MLP input at r.time=t
# # - where N is the number of op
# # - remember MLP inputs are indicated by 0,-1,-2 etc. in r.inputs
# # We only keep the copies necessary for the back calculation
# # - initstack initializes necessary r.stack[n] with empty arrays
# #   the other elements in r.stack are left unassigned
# # - allocstack makes sure at least r.time arrays are allocated
# #   for each assigned r.stack[n]
# #   allocstack also copies inputs x as necessary

# function initstack(r::Net)
#     N = length(r.op)
#     hist = Array(Any,N+ninputs(r))
#     for n=1:N
#         back_reads_y(r.op[n]) && !isassigned(hist,n) && (hist[n]=Any[])
#         if back_reads_x(r.op[n])
#             for i in r.inputs[n]
#                 i <= 0 && (i = N+1-i)
#                 !isassigned(hist,i) && (hist[i]=Any[])
#             end
#         end
#     end
#     return hist
# end

# function allocstack(r::Net, x)
#     N = length(r.op)
#     for n=1:N
#         isassigned(r.stack,n) || continue
#         length(r.stack[n]) >= r.time && continue
#         length(r.stack[n]) == r.time - 1 || error("Stack corruption")
#         push!(r.stack[n], KUdense(atype(x[1]), eltype(x[1]), (0,)))
#     end
#     for i=1:length(x)
#         n=N+i
#         isassigned(r.stack,n) || continue
#         if length(r.stack[n]) < r.time
#             length(r.stack[n]) == r.time - 1 || error("Stack corruption")
#             push!(r.stack[n], copy(x[i]))
#         else
#             copy!(r.stack[n][r.time], x[i])
#         end
#     end
# end


# function forw1(r::Net, x; train=true, o...)
#     train ? 
#     forwtrain(r, x; o...) :
#     forwpredict(r, x; o...)
# end

# function forwtrain(r::Net, x; o...)
#     r.t += 1
#     copyto(r.x, x, r.t)
#     for n = 1:N
#         r.y[n] = forw(r.net[n], forwinput(r,n,x)...; o...)
#         if !predict

#             if ((n == N) ||                     # preserve the last layer
#                 (length(r.freads[n]) > 1) ||    # preserve if multiple forw reads
#                 (length(r.breads[n]) > 0))      # preserve if any back reads
#                 copyto(r.h, h, n, r.t)
#             else
#                 r.h[n,r.t] = h
#             end
#         end
#         return r.y[N]
#     end
# end


# function copyto(a, x, i...)
#     # resize a if necessary here...
#     ai = getindex(a, i...)
#     if ai == nothing
#         setindex!(a, x, i...)
#     else
#         resize!(ai, size(x))
#         copy!(ai, x)
#     end
#     return getindex(a, i...)
# end


# # DONE: the semantics of forw should be simpler: each unit reads the
# # values of its inputs as they are at that point in time, whether they
# # are ahead in the network or behind.


# function inithidden(r::Net, x, N, T)
#     if !isdefined(r,x)
#         r.(x) = fill!(Array(Any, N, T), nothing)
#     elseif size(r.(x), 1) != N
#         error("Size mismatch")
#     elseif size(r.(x), 2) != T
#         isa(r.(x), SubArray) && (r.(x) = r.(x).parent)
#         if size(r.(x), 2) > T
#             r.(x) = sub(r.(x), 1:N, 1:T)
#         elseif size(r.(x), 2) < T
#             h = Array(Any, N, T)
#             copy!(h, 1, r.(x), 1, length(r.(x)))
#             h[length(r.(x))+1:end] = nothing
#             r.(x) = h
#         end
#     end
#     @assert size(r.(x)) == (N,T)
#     return r.(x)
# end




# OK: pass x,y 
## OK: To pass x, y take a look at the current interface
## OK: Return dx and param x for back may be tuples
# OK: figure out dw increment
## OK: dw gets zeroed once in the beginning
## OK: incr=(t<T): what if at t==T dh was nothing so it didn't get zeroed out?
# OK: figure out dx increment
## OK: To do dx increment, rnn back should handle it
## OK: what if we copy same dx to two inputs and one wants to increment
## OK: dy for layers with forwreads > 1 needs to be zeroed every time step
## NO: it needs to be zeroed by its last output, which is the largest index smaller than it, or if not largest index in forwreads: remember that and zero with that, check reset array and if target nonempty increment or reset.
## OK: actually it can be just zeroed as soon as it is used.

# OK: Initialize w the first time?
# OK: back needs x, y, and incr options.
# OK: get rid of forw.y and back.dx parameters

# OK: dy should be like x, only t component, and refer to the output of net[N,t]
# OK: if dy[t] is nothing do not propagate any error.
# OK: we dont want to alloc dy per time step, we want one dy per layer, let layers manage them.

# function rnnget(rnn, x, y, t, l)
#     isa(rnn[l],Tuple) ?
#     rnnget(rnn, x, y, t, l, rnn[l][1], rnn[l][2:end]) :
#     rnnget(rnn, x, y, t, l, rnn[l], tuple((l-ninputs(rnn[l])):(l-1)...))
# end

# function rnnget(rnn, x, y, t, l, layer, index)
#     n = ninputs(layer)          # DONE: ninputs not defined (done for ninputs=1 default)
#     length(index) == n || error("Wrong number of inputs")
#     input = map(index) do i
#         i == 0 ? x[t] :
#         i < l  ? y[i,t] :
#         y[i,t-1]  # DONE: problem with t=1
#     end
#     n == 1 && (input = input[1])
#     return (layer, input)
# end

# function rnninit(rnn, x)        # DONE: avoid allocation every call: have an rnn type?
#     # the first row is reserved for the input
#     # the first column is reserved for empty arrays?
#     # or should we try to accept nothing input?
#     # but what about batch size?  should be hdim x nbatch
#     # hinton says train h0 like the weights: copy column nbatch times
#     y = fill!(Array(Any, 1+length(rnn), 1+length(x)), nothing)
# end

# nothings(m,n)=fill!(Array(Any,m,n),nothing)

# function forwreads(r::Net)
#     N = length(r.inputs)
#     freads = [ Int[] for n=1:N ]
#     for n=1:N
#         for x in r.inputs[n]
#             (x>0) && push!(freads[x], n)
#         end
#     end
#     freads
# end

# function backreads(r::Net)
#     N = length(r.inputs)
#     freads = forwreads(r)
#     breads = [ Int[] for n=1:N ]
#     for n=1:N
#         for o in freads[n]
#             back_reads_x(r.net[o]) && push!(breads[n], o)
#         end
#         back_reads_y(r.net[n]) && push!(breads[n], n)
#     end            
#     breads
# end

# function rnnsaves(r::Net)
#     map(forwreads(r), backreads(r)) do f,b
#         length(f) > 1 || length(b) > 0
#     end
# end

# type RNN1; net; inputs; freads; breads; t; x; y; Y; dy;
#     function RNN1(a...)
#         r = new()
#         n = length(a)
#         r.net = map(ai->(isa(ai,Tuple) ? ai[1] : ai), a)
#         r.inputs = [ Int[] for i=1:n ] # inputs[j]={i: h[j,t] needs h[i,t']}
#         # DONE: we only need the counts not the elements for freads and breads
#         r.freads = [ Int[] for i=1:n ] # freads[i]={j: h[j,t] needs h[i,t']}
#         r.breads = [ Int[] for i=1:n ] # breads[i]={j:dy[j,t] needs h[i,t']} 
#         for j=1:n
#             inputs = isa(a[j],Tuple) ? a[j][2:end] : ((j-ninputs(a[j])):(j-1))
#             ninputs(r.net[j]) == length(inputs) || error("Wrong number of inputs")
#             overwrites(r.net[j]) && in(0, inputs) && error("$j:$(typeof(r.net[j])) overwrites Net input")
#             push!(r.inputs[j], inputs...)
#             for i in r.inputs[j]
#                 (0 <= i <= n) || error("Bad input index")
#                 (i > 0) && push!(r.freads[i], j)
#             end
#         end
#         # DONE: there is no need for this separate for loop
#         for i=1:n
#             for j in r.freads[i]
#                 back_reads_x(r.net[j]) && push!(r.breads[i], j)
#             end
#             back_reads_y(r.net[i]) && push!(r.breads[i], i)
#         end
#         r.t = 0
#         return r
#     end            
# end


# # NO: merge x, h?  x has too many special exceptions.
# # DONE: apply the same dependency analysis to the first layer and the last layer too.

# function forwinput(r::Net, x, n)
#     indices = r.inputs[n]
#     arrays = Array(Any, length(indices))
#     for j = 1:length(indices)
#         i = indices[j]
#         arrays[j] = ((i == 0) ?
#                      (overwrites(r.net[n],j) ? copyto(r.y, x, n) : x) :
#                      (overwrites(r.net[n],j) && (length(r.freads[i])>1) ? copyto(r.y, r.y[i], n) : r.y[i]))
#     end
#     return arrays
# end

# copy or point to x from output[n+1]?
# how to handle t=0
# NO: have r.inputs contain the actual matrices? sometimes inputs are "nothing"
# ASSUME: do we still need protection from overwrites or should the compiler handle that?
# OK: all op need to support y keyword option.
# DONE: make sure the y option is not treated optional!

        # if !predict && isassigned(r.stack,n)
        #     if r.output[n] != nothing
        #         copy!(r.stack[n][r.time], r.output[n])
        #     else
        #         warn("Recording 'nothing' in stack at $n:$(typeof(r.op[n]))")
        #         r.stack[n][r.time] = nothing
        #     end
        # end
#     # incr w
#     # incr dy
#     # multiple dy
#     # storage for dy: initalization, increment, sharing?
#     # cannot use the same registers as the output, may need simultaneously, may have different sharing
#     # pop for previous reg values
#     # r.dy[n] for diff registers





# function forwinput(r::Net, x, n)
#     map(r.inputs[n]) do i               # return the input matrix or tuple of input matrices to produce r.output[n]
#         i <= 0     ? x[1-i] :           # i <= 0 is used to indicate input x[1-i]
#         i < n      ? r.output[i] :      # 0 < i < n are outputs from the current time slice
#         r.time > 1 ? r.output[i] :	# i >= n are from the previous time slice if r.time > 1
#         nothing                         # nothing represents zero matrix from r.time=0
#     end
# end

# DONE: pushinput takes a lot of time, do we need it for ffnn?  
# Yes, we don't know if the next move is forw or back.

# DONE: l.y and l.dx for layers should point to internal storage, not externally provided ones.
# right now an op may not use the provided y as a result of resize.
# ops should not resize externally provided output arrays? (at least without a warning)

# DONE: r.dy: last op has an extra output to the outside!
# DONE: r.dy: what to do with the network inputs: alloc for now but do not use makes code simpler.
# DONE: r.dy: implement dx option for back for all ops

# DONE: add reset (reset taken, called it init)

# # registers can be (1) nothing, (2) pointer to net input, (3) pointer to net.buf

# regbuf(r,n)=(r.reg[r.out[n]]!=nothing ? r.reg[r.out[n]] : r.buf[r.out[n]])

# # r.reg[i] is the i'th register of the MLP
# # Each register points to an array allocated elsewhere 
# # or is set to 'nothing' representing the zero matrix
# # These registers are used for network inputs, op outputs, op gradients.
# # initreg(r) just creates an empty array of registers r.reg
# # other init functions below add registers as needed to r.reg

# function initreg(r::Net)
#     r.reg = Any[]
# end

# function initbuf(r::Net)
#     r.buf = copy(r.reg) # array of nothings, initialized at initforw
# end

# function initregi(r::Net, i::Int, y, dims=size(y))
#     dy = r.reg[i]
#     dy == nothing && (dy = r.buf[i])
#     if dy == nothing
#         dy = r.buf[i] = similar(y, dims)
#     end
#     if size(dy) != dims
#         warn("Resizing $(size(dy))->$dims")
#         dy = r.buf[i] = resize!(dy,dims)
#     end
#     r.reg[i] == dy || r.reg[i] == nothing || error("reg $i mismatch")
#     atype(dy) == atype(y) || error("atype mismatch")
#     eltype(dy) == eltype(y) || error("eltype mismatch")
#     r.reg[i] = dy
# end

        # x = xx[1]
        # i = r.out[n]
        # y = getybuf(r,n)
        # if y == nothing
        #     y = r.buf[i] = similar(x, s)
        # end
        # if size(y) != s
        #     warn("Resizing $(size(y))->$s")
        #     y = r.buf[i] = resize!(y,s)
        # end
        # r.reg[i] == y || r.reg[i] == nothing || error("reg mismatch")
        # atype(y) == atype(x) || error("atype mismatch")
        # eltype(y) == eltype(x) || error("eltype mismatch")
            # DONE: (back) buf gets the result but reg is still nothing? no need for buf/reg distinction in back
            # DONE: (back) zero out the dw
# op(r::Net,n::Int)=r.op[n]

# gety(r::Net,n::Int)=r.reg[r.out[n]]
# getdy(r::Net,n::Int)=r.reg[r.dif[n]]
# getx(r::Net,n::Int)=r.reg[r.out[r.inputs[n]]]
# getdx(r::Net,n::Int)=r.reg[r.dif[r.inputs[n]]]

# getbuf(r::Net,i::Int)=(r.reg[i]!=nothing?r.reg[i]:r.buf[i])
# getybuf(r::Net,n::Int)=getbuf(r,r.out[n])
# getdybuf(r::Net,n::Int)=getbuf(r,r.dif[n])
# getxbuf(r::Net,n::Int)=map(i->getybuf(r,i),r.inputs[n])
# getdxbuf(r::Net,n::Int)=map(i->getdybuf(r,i),r.inputs[n])

# get1x(r::Net,n::Int)=get1(getx(r,n))
# get1dxbuf(r::Net,n::Int)=get1(getdxbuf(r,n))

# sety(r::Net,n::Int,x)=(r.reg[r.out[n]]=x)
# setdy(r::Net,n::Int,x)=(r.reg[r.dif[n]]=x)
# setinput(r::Net,n::Int,x)=(r.reg[r.out[n+nops(r)]]=x)
# getinput(r::Net,n::Int)=r.reg[r.out[n+nops(r)]]

# pushy(r::Net,n::Int)=(r.save[n] && pushreg(r,r.out[n]))
# popy(r::Net,n::Int)=(r.save[n] ? popreg(r,r.out[n]) : r.reg[r.out[n]])
# pushinput(r::Net,n::Int)=(r.save[n+nops(r)] && pushreg(r,r.out[n+nops(r)]))
# popinput(r::Net,n::Int)=(r.save[n+nops(r)] && popreg(r,r.out[n+nops(r)]))

# function pushreg(r::Net,i::Int)
#     length(r.stack) <  r.sp && error("Stack error")
#     length(r.stack) == r.sp && push!(r.stack, :newcell)
#     r.sp += 1
#     if r.reg[i] == nothing
#         r.stack[r.sp] == nothing || r.stack[r.sp] == :newcell || warn("pushing nothing")
#         r.stack[r.sp] = nothing
#     elseif r.stack[r.sp] == nothing
#         warn("copying over nothing")
#         r.stack[r.sp] = copy(r.reg[i])
#     elseif r.stack[r.sp] == :newcell
#         r.stack[r.sp] = copy(r.reg[i])
#     elseif size(r.reg[i]) != size(r.stack[r.sp])
#         warn("resizing during push")
#         copy!(r.stack[r.sp], r.reg[i])
#     else
#         copy!(r.stack[r.sp], r.reg[i])
#     end
# end

# function popreg(r::Net,i::Int)
#     r.sp > 0 || error("Stack error")
#     if r.reg[i] == nothing
#         r.stack[r.sp] == nothing || warn("popping array over nothing")
#     elseif r.stack[r.sp] == nothing
#         # warn("popping nothing over array")
#     elseif size(r.reg[i]) != size(r.stack[r.sp])
#         warn("resizing during pop")
#     end
#     r.reg[i] = r.stack[r.sp]
#     r.sp -= 1
#     return r.reg[i]
# end

# function initsequence(r::Net, x::Vector; trn=false, a...)
#     r.sp == 0 || error("Stack corruption")
#     inputs = isa(x[1],Tuple) ? x[1] : (x[1],)
#     initbatch(r, inputs...; trn=trn, seq=true, a...)
#     fill!(r.out, nothing)                               # to represent zero matrices at t=0
#     if trn
#         fill!(r.dif, nothing)                           # why? (TODO)
#         for n=1:length(r.dif0)
#             r.multi[n] && fill!(r.dif0[n], 0)           # zeroed by back at every item in sequence
#         end
#         for w in params(r)
#             fill!(w.diff, 0)                            # zeroed only once at the beginning of the sequence
#         end
#     end
# end

# function initbatch(r::Net, inputs...; trn=false, seq=false)
#     length(inputs) == ninputs(r) || error("Wrong number of inputs")
#     initout0(r, inputs...)
#     !seq && fill!(r.out, nothing)
#     if trn
#         initparams(r, inputs...; seq=seq)
#         initdif0(r)
#         if !seq
#             fill!(r.dif, nothing)                           # why? (TODO)
#             for n=1:length(r.dif0)
#                 r.multi[n] && fill!(r.dif0[n], 0)           # zeroed by back at every item in sequence
#             end
#             for w in params(r)
#                 fill!(w.diff, 0)                            # zeroed only once at the beginning of the sequence
#             end
#         end
#     end
# end
    
# DONE: incremental updates
 # DONE: forw: do we really need inputs to be tuple here? YES, have the option of multi-input nets like multi-input ops.
# DONE: forw: (minor) this ends up using a lot of nothing pushes first minibatch, that's ok, 'nothing' is a legit value.

# Terminology:
# An instance is an item or a sequence.
# An item is a contiguous feature array: x[d...]
# A sequence is a sequence of items in time: x[t][d...]
# An item batch is a contiguous array with an extra dimension: x[d...,i]
# inputs[i] should be an item batch.

# function initbatch(r::Net, inputs...)
    
# end

# function init(r::Net, x::Vector; a...)
#     if isempty(x) || x[1] == nothing
#         error("Got nothing as input")
#     elseif isa(x[1], Tuple)
#         init(r, x[1]...; seq=true, a...)
#     elseif isbits(eltype(x[1]))
#         init(r, x[1]; seq=true, a...)
#     end
# end

# DONE:
# + try the adding problem with irnn
# + gradient analysis with adding, should fail
# + add incremental update of dy and confirm gradient testing is now ok
# + let forw and back write the input to their own registers
# x ops with forw=nothing should accept back=nothing input
# + minibatching
# x rethink the 'nothing' optimization, removed from dif but not out?
# + return value for back, returndx option: no need, return if dx specified.
# + add gradient clipping
# + implement train/predict and try ffnn mnist experiments: how do we treat sequences and minibatches?
# + we need to solve predict() and accuracy() problems first: use MLP for now?

### initforw:

# We have a batch or a sequence of batches as input.
# The batch could be for an FNN or an RNN.
# Before each forw-batch we make sure out0 has the right size.
# Before each back-batch we make sure dif0 has the right size.
# dw needs to be zeroed only at the beginning of a sequence.
# dw does not need incr if not part of a sequence.
# out and dif registers need to be nothinged only at the beginning of a sequence.
# we need to preserve their content from one rnn-batch to the next.
# dy for multi ops are zeroed during back so init should not worry.
# so we need to know if we are in a sequence or not: add a keyword arg to forw and back.

# The input x::Vector can be x[i][t][d...,b] or x[t][d...,b]
# where i:instance, t:time, d:dims, b:batch
# We only want to init if x[t][d...,b]
# In that case x[1] will be a numeric array 
# or a tuple of numeric arrays (to support multiple inputs)

# init before a sequence zeroes out everything:
# out[:] is set to nothing to represent zero matrices at t=0
# dif[:] is set to nothing why?  (TODO)
# dif0[n] is set to zero for r.multi[n]
# but first we need to allocate using the batch init
# the input is x[t][d...,i] for single input
# or x[t][input][d...,i] for multiple inputs

# at the beginning of a sequence:
# we want out[:]=nothing, dif[:]=nothing, dw=0, dif0[i] for multi.
# before every itembatch:
# we just want to fix the sizes.

"""
Life cycle of registers: 
	
		out	out0	dif	dif0	dif1	dw	stack	calls
Net		-	-	-	-	-	-	-	initout,initdif
initout		n	[]	-	-	-	-	-	
initdif		-	-	n	[]	[]+m	-	-	
 forwseq	-	-	-	-	-	-	-	initseq,forw+s*
  initseq	n	-	n+t	0+t+m	-	0+t	-	initbat
 forw		wr	w	-	-	-	-	w	initbat,seq passed to initbat
  initbat	n-s	-	n+t-s	0+t-s+m	-	0+t-s	-	initpar2(trn),initdif0(trn),initout0,seq passed to initpar2
   initout0	-	a	-	-	-	-	-	
   initpar2	-	r	-	-	-	a	-	
   initdif0	-	r	-	a	a+m	-	-
 backseq	-	-	-	-	-	-	-	back+s*
  back		rw(pop)	-	nwr0	w	w	-	r	seq passed as incr to op.back
 loss		r	-	-	-	-	-	-	

(+m:multi, +s:seq, +t:trn, r:read, w:write, a:alloc, 0:zero-out, n:fill-nothing)
"""
