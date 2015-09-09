type RNN; op; inputs; ninputs; save; reg; y; dy; stack; sp; buf; 
    function RNN(a...)
        r = new()
        initop(r, a...)
        initinputs(r, a...)
        @assert length(r.op)==length(r.inputs)
        initninputs(r)
        initreg(r)
        initsave(r)
        inity(r)
        initdy(r)
        initstack(r)
        initbuf(r)
        return r
    end
end

function forw(r::RNN, inputs...; train=true, a...)
    length(inputs) == ninputs(r) || error("Wrong number of inputs")
    for i = 1:ninputs(r)
        train && pushinput(r,i)
        setinput(r,i,inputs[i])
        println("in[$i]=$(map(ptr16,inputs)) st=$(map(ptr16,r.stack[1:r.sp]))")
    end
    initforw(r)
    for n = 1:nops(r)
        train && pushy(r,n)
        sety(r,n,forw(op(r,n), getx(r,n)...; y=getybuf(r,n), a...))
        println("op[$n]:$((typeof(op(r,n)),map(ptr16,getx(r,n))...,ptr16(gety(r,n)))) st=$(map(ptr16,r.stack[1:r.sp]))")
    end
    gety(r,nops(r))
end


function back(r::RNN, dy; a...)
    setdy(r,nops(r),dy)
    println("back:dy=$((ptr16(getdy(r,nops(r))),))")
    initback(r)
    for n = nops(r):-1:1
        if getdy(r,n) != nothing        # 'nothing' represents 0 loss gradient
            # TODO: returndx=false when we reach network input
            # TODO: incremental shit
            # DONE: buf gets the result but reg is still nothing? no need for buf/reg distinction in back
            # TODO: zero out the dw
            back(op(r,n), getdy(r,n); incr=true, x=get1x(r,n), y=gety(r,n), dx=get1dxbuf(r,n), a...)
        end
        println("op[$n]:$((typeof(op(r,n)),:x,map(ptr16,getx(r,n))...,:y,ptr16(gety(r,n)),:dy,ptr16(getdy(r,n)),:dx,map(ptr16,getdxbuf(r,n))...)) st=$(map(ptr16,r.stack[1:r.sp]))")
        popy(r,n)
        println("pop[$n]:y=$((ptr16(gety(r,n)),)) st=$(map(ptr16,r.stack[1:r.sp]))")
    end
    for i = ninputs(r):-1:1
        println("in[$i]=$(map(ptr16,(getinput(r,i),))) st=$(map(ptr16,r.stack[1:r.sp]))")
        popinput(r,i)
        println("in[$i]=$(map(ptr16,(getinput(r,i),))) st=$(map(ptr16,r.stack[1:r.sp]))")
    end
end

ninputs(r::RNN)=r.ninputs
nops(r::RNN)=length(r.op)
op(r::RNN,n::Int)=r.op[n]

gety(r::RNN,n::Int)=r.reg[r.y[n]]
getdy(r::RNN,n::Int)=r.reg[r.dy[n]]
getx(r::RNN,n::Int)=r.reg[r.y[r.inputs[n]]]
getdx(r::RNN,n::Int)=r.reg[r.dy[r.inputs[n]]]

getbuf(r::RNN,i::Int)=(r.reg[i]!=nothing?r.reg[i]:r.buf[i])
getybuf(r::RNN,n::Int)=getbuf(r,r.y[n])
getdybuf(r::RNN,n::Int)=getbuf(r,r.dy[n])
getxbuf(r::RNN,n::Int)=map(i->getybuf(r,i),r.inputs[n])
getdxbuf(r::RNN,n::Int)=map(i->getdybuf(r,i),r.inputs[n])

get1(x)=(length(x)==1?x[1]:x)
get1x(r::RNN,n::Int)=get1(getx(r,n))
get1dxbuf(r::RNN,n::Int)=get1(getdxbuf(r,n))

sety(r::RNN,n::Int,x)=(r.reg[r.y[n]]=x)
setdy(r::RNN,n::Int,x)=(r.reg[r.dy[n]]=x)
setinput(r::RNN,n::Int,x)=(r.reg[r.y[n+nops(r)]]=x)
getinput(r::RNN,n::Int)=r.reg[r.y[n+nops(r)]]

pushy(r::RNN,n::Int)=(r.save[n] && pushreg(r,r.y[n]))
popy(r::RNN,n::Int)=(r.save[n] ? popreg(r,r.y[n]) : r.reg[r.y[n]])
pushinput(r::RNN,n::Int)=(r.save[n+nops(r)] && pushreg(r,r.y[n+nops(r)]))
popinput(r::RNN,n::Int)=(r.save[n+nops(r)] ? popreg(r,r.y[n+nops(r)]) : error("Input $n not saved"))

function pushreg(r::RNN,i::Int)
    length(r.stack) <  r.sp && error("Stack error")
    length(r.stack) == r.sp && push!(r.stack, :newcell)
    r.sp += 1
    if r.reg[i] == nothing
        r.stack[r.sp] == nothing || r.stack[r.sp] == :newcell || warn("pushing nothing")
        r.stack[r.sp] = nothing
    elseif r.stack[r.sp] == nothing
        warn("copying over nothing")
        r.stack[r.sp] = copy(r.reg[i])
    elseif r.stack[r.sp] == :newcell
        r.stack[r.sp] = copy(r.reg[i])
    elseif size(r.reg[i]) != size(r.stack[r.sp])
        warn("resizing during push")
        copy!(r.stack[r.sp], r.reg[i])
    else
        copy!(r.stack[r.sp], r.reg[i])
    end
end

function popreg(r::RNN,i::Int)
    r.sp > 0 || error("Stack error")
    if r.reg[i] == nothing
        r.stack[r.sp] == nothing || warn("popping array over nothing")
    elseif r.stack[r.sp] == nothing
        warn("popping nothing over array")
    elseif size(r.reg[i]) != size(r.stack[r.sp])
        warn("resizing during pop")
    end
    r.reg[i] = r.stack[r.sp]
    r.sp -= 1
    return r.reg[i]
end

# TODO: l.y and l.dx for layers should point to internal storage, not externally provided ones.
# right now an op may not use the provided y as a result of resize.
# ops should not resize externally provided output arrays? (at least without a warning)

### Net initialization

# r.op[n] is the n'th operation in the net
# The user specifies the operations in the Net constructor arguments
# If an argument is another Net, its operations are spliced in

function initop(r::RNN, a...)
    r.op = Layer[]
    for ai in a
        isa(ai,Tuple) && (ai=ai[1])
        isa(ai,Layer) ?  push!(r.op, ai) :
        isa(ai,RNN)   ?  append!(r.op, ai.op) :
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

function initinputs(r::RNN, a...)
    newindex = Array(Int, length(a))
    lastindex = 0
    for i=1:length(a)
        ai = isa(a[i],Tuple) ? a[i][1] : a[i]
        lastindex += (isa(ai,Layer) ? 1 : length(ai.op))
        newindex[i] = lastindex
    end
    r.inputs = Any[]
    for i=1:length(a)
        ai = isa(a[i],Tuple) ? a[i][1] : a[i]
        bi = isa(a[i],Tuple) ? a[i][2:end] : ((i-ninputs(ai)):(i-1))
        length(bi) == ninputs(ai) || error("Wrong number of inputs for $i:$(typeof(ai))")
        if isa(ai, Layer)
            push!(r.inputs, map(j->(j>0 ? newindex[j] : nops(r)+1-j), Int[bi...]))
        else
            j0 = length(r.inputs)
            for aii in ai.inputs
                push!(r.inputs, map(j->(j<=nops(ai) ? j+j0 : (j=bi[j-nops(ai)]; j>0 ? newindex[j] : nops(r)+1-j)), aii))
            end
        end
    end
end

# r.ninputs is the number of inputs the whole Net expects
# indices i>nops(r) in r.inputs refer to network inputs

function initninputs(r::RNN)
    n = 0
    for ai in r.inputs
        for aij in ai
            aij - nops(r) > n && (n = aij - nops(r))
        end
    end
    r.ninputs = n
end

# r.save[n] is true if the result of op[n] (for n <= nops(r))
# or the network input n-nops(r) (for n > nops(r))
# should be saved for back calculation

function initsave(r::RNN)
    r.save = falses(nops(r)+ninputs(r))
    for n=1:nops(r)
        back_reads_y(r.op[n]) && (r.save[n] = true)
        if back_reads_x(r.op[n])
            for i in r.inputs[n]
                r.save[i] = true
            end
        end
    end
end

# r.reg[i] is the i'th register of the Net
# Each register points to an array allocated elsewhere 
# or is set to 'nothing' representing the zero matrix
# These registers are used for network inputs, op outputs, op gradients.
# initreg(r) just creates an empty array of registers r.reg
# other init functions below add registers as needed to r.reg

function initreg(r::RNN)
    r.reg = Any[]
end

# r.y[n] is the index of the register that holds the output of op[n] (for n<=nops(r))
# or the network input n-nops(r) (for n>nops(r))
# we optimize register use by overwriting existing ones when we can

function inity(r::RNN)
    r.y = zeros(Int,nops(r)+ninputs(r))
    index = zeros(Int,nops(r))
    for n=1:nops(r)
        r.save[n] && continue           # a saved register should only be written by op[n]
        if overwrites(r.op[n])
            i = r.inputs[n][1]          # see if we can overwrite the first input
            r.save[i] && continue       # do not overwrite if you are going to save for back
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
    for n=1:ninputs(r)                  # use the first registers for network inputs
        push!(r.reg,nothing)
        r.y[n+nops(r)] = length(r.reg)
    end
    for n=1:nops(r)
        index[n]==0 || continue         # index==0 represents need for new register
        push!(r.reg,nothing)
        r.y[n] = length(r.reg)
    end
    for n=1:nops(r)                     # other ops will overwrite existing registers
        index[n]==0 && continue
        k = index[n]
        while index[k]!=0; k=index[k]; end
        r.y[n] = r.y[k]
    end
end

# r.dy[n] is the index of the register that holds the loss gradient of the last output of op[n] (for n<=nops(r))
# TODO: last op has an extra output to the outside!
# TODO: what to do with the network inputs
# DONE: implement dx option for back for all ops
# TODO: this is not optimal for add: no need for two copies of dy when neither input is overwriting.

function initdy(r::RNN)
    ny = nops(r)+ninputs(r)             # 1..nops(r) for op outputs, nops(r)+1..nops(r)+ninputs(r) for network inputs
    r.dy = zeros(Int,ny)
    noutputs = zeros(Int,ny)
    noutputs[nops(r)] = 1               # for the network output
    for n=1:nops(r)
        for i in r.inputs[n]
            noutputs[i] += 1
        end
    end
    index = zeros(Int,ny)
    for n=1:nops(r)                     # should back(op[n]) overwrite dy[n]?
        overwrites(r.op[n]) || continue # overwrites means potentially both forw x<-y and back dy<-dx
        noutputs[n] > 1 && continue     # don't share dy[n] with multi-output
        for i in r.inputs[n]
            noutputs[i] > 1 && continue # don't share dy[n] with multi-output
            index[i]==0 || error("should not happen")
            index[i]=n; break           # op[n] will overwrite dy[n] to get dy[i]
        end
    end
    for n=1:ny
        if index[n]==0                  # index==0 represents need for new register
            push!(r.reg,nothing)
            r.dy[n] = length(r.reg)
        end
    end
    for n=1:ny
        if index[n] > 0
            k = index[n]
            while index[k]!=0; k=index[k]; end
            r.dy[n] = r.dy[k]
        end
    end
end

function initstack(r::RNN)
    r.stack = Any[]
    r.sp = 0
end

function initbuf(r::RNN)
    r.buf = copy(r.reg) # array of nothings, initialized at initforw
end

# registers can be (1) nothing, (2) pointer to net input, (3) pointer to net.buf

regbuf(r,n)=(r.reg[r.y[n]]!=nothing ? r.reg[r.y[n]] : r.buf[r.y[n]])

function initforw(r::RNN)
    for n = 1:nops(r)
        xx = getxbuf(r,n)
        s = ysize(op(r,n), xx...)
        s == nothing && continue
        x = xx[1]
        i = r.y[n]
        y = getybuf(r,n)
        if y == nothing
            y = r.buf[i] = similar(x, s)
        end
        if size(y) != s
            warn("Resizing $(size(y))->$s")
            y = r.buf[i] = resize!(y,s)
        end
        r.reg[i] == y || r.reg[i] == nothing || error("reg mismatch")
        atype(y) == atype(x) || error("atype mismatch")
        eltype(y) == eltype(x) || error("eltype mismatch")
    end
end

ptr16(x)=(x==nothing ? UInt16(0) : UInt16(Int(pointer(x)) % 65521))
ptr8(x)=(x==nothing ? UInt8(0) : UInt8(Int(pointer(x)) % 251))

function initback(r::RNN)
    initdiff(r)
    for n = nops(r):-1:1
        getdy(r,n) == nothing && continue
        for i in r.inputs[n]
            y = gety(r,i)
            y == nothing && error("Lost y")
            j = r.dy[i]
            dy = getdybuf(r,i)
            if dy == nothing 
                dy = r.buf[j] = similar(y)
            end
            if size(dy) != size(y)
                warn("Resizing $(size(dy))->$(size(y))")
                dy = r.buf[j] = resize!(dy,size(y))
            end
            r.reg[j] == dy || r.reg[j] == nothing || error("reg $j mismatch")
            atype(dy) == atype(y) || error("atype mismatch")
            eltype(dy) == eltype(y) || error("eltype mismatch")
            r.reg[j] = dy
        end
    end
end

function initdiff(r::RNN)
    for l in r.op
        w = param(l)
        w == nothing && continue
        similar!(w, :diff, w.arr)
        similar!(w, :inc, w.arr)
        fill!(w.diff, 0)
    end
end

### DEAD CODE

# function back(r::RNN, dy; o...)                 # dy[1:T] loss gradients for output y
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

# # r.buf[n] is the array pre-allocated for register r.reg[n]
# # r.buf is initialized in three steps:
# # - initbuf determines the indices for array reuse.
# #   i=0 means allocate, i>0 means reuse r.buf[i]
# # - allocbuf allocates 0 size KUdense arrays
# #   this is a separate step because we need input x for type info
# # - the actual op resize the bufs as needed.

# # TODO: this should also look at back needs when allocating

# function initbuf(r::RNN)
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

# function allocbuf(r::RNN, x)
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

# function initoutput(r::RNN)
#     r.output = fill!(Array(Any,length(r.op)), nothing)
# end

# # r.stack[n<=N][t] is a copy of the output of r.op[n] at r.time=t
# # r.stack[n>N][t] is a copy of the n-N'th Net input at r.time=t
# # - where N is the number of op
# # - remember Net inputs are indicated by 0,-1,-2 etc. in r.inputs
# # We only keep the copies necessary for the back calculation
# # - initstack initializes necessary r.stack[n] with empty arrays
# #   the other elements in r.stack are left unassigned
# # - allocstack makes sure at least r.time arrays are allocated
# #   for each assigned r.stack[n]
# #   allocstack also copies inputs x as necessary

# function initstack(r::RNN)
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

# function allocstack(r::RNN, x)
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


# function forw1(r::RNN, x; train=true, o...)
#     train ? 
#     forwtrain(r, x; o...) :
#     forwpredict(r, x; o...)
# end

# function forwtrain(r::RNN, x; o...)
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
#     # TODO resize a if necessary here...
#     ai = getindex(a, i...)
#     if ai == nothing
#         setindex!(a, x, i...)
#     else
#         resize!(ai, size(x))
#         copy!(ai, x)
#     end
#     return getindex(a, i...)
# end


# # TODO: the semantics of forw should be simpler: each unit reads the
# # values of its inputs as they are at that point in time, whether they
# # are ahead in the network or behind.


# function inithidden(r::RNN, x, N, T)
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



# TODO: figure out when no back needed
## if the previous layer does not need it (rand, input) do not propagate. TODO: how to detect?

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
#     n = ninputs(layer)          # TODO: ninputs not defined (done for ninputs=1 default)
#     length(index) == n || error("Wrong number of inputs")
#     input = map(index) do i
#         i == 0 ? x[t] :
#         i < l  ? y[i,t] :
#         y[i,t-1]  # TODO: problem with t=1
#     end
#     n == 1 && (input = input[1])
#     return (layer, input)
# end

# function rnninit(rnn, x)        # TODO: avoid allocation every call: have an rnn type?
#     # the first row is reserved for the input
#     # the first column is reserved for empty arrays?
#     # or should we try to accept nothing input?
#     # but what about batch size?  should be hdim x nbatch
#     # hinton says train h0 like the weights: copy column nbatch times
#     y = fill!(Array(Any, 1+length(rnn), 1+length(x)), nothing)
# end

# nothings(m,n)=fill!(Array(Any,m,n),nothing)

# function forwreads(r::RNN)
#     N = length(r.inputs)
#     freads = [ Int[] for n=1:N ]
#     for n=1:N
#         for x in r.inputs[n]
#             (x>0) && push!(freads[x], n)
#         end
#     end
#     freads
# end

# function backreads(r::RNN)
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

# function rnnsaves(r::RNN)
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
#         # TODO: we only need the counts not the elements for freads and breads
#         r.freads = [ Int[] for i=1:n ] # freads[i]={j: h[j,t] needs h[i,t']}
#         r.breads = [ Int[] for i=1:n ] # breads[i]={j:dy[j,t] needs h[i,t']} 
#         for j=1:n
#             inputs = isa(a[j],Tuple) ? a[j][2:end] : ((j-ninputs(a[j])):(j-1))
#             ninputs(r.net[j]) == length(inputs) || error("Wrong number of inputs")
#             overwrites(r.net[j]) && in(0, inputs) && error("$j:$(typeof(r.net[j])) overwrites RNN input")
#             push!(r.inputs[j], inputs...)
#             for i in r.inputs[j]
#                 (0 <= i <= n) || error("Bad input index")
#                 (i > 0) && push!(r.freads[i], j)
#             end
#         end
#         # TODO: there is no need for this separate for loop
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
# # TODO: apply the same dependency analysis to the first layer and the last layer too.

# function forwinput(r::RNN, x, n)
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
# TODO: make sure the y option is not treated optional!

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





# function forwinput(r::RNN, x, n)
#     map(r.inputs[n]) do i               # return the input matrix or tuple of input matrices to produce r.output[n]
#         i <= 0     ? x[1-i] :           # i <= 0 is used to indicate input x[1-i]
#         i < n      ? r.output[i] :      # 0 < i < n are outputs from the current time slice
#         r.time > 1 ? r.output[i] :	# i >= n are from the previous time slice if r.time > 1
#         nothing                         # nothing represents zero matrix from r.time=0
#     end
# end

# function reset(r::RNN)
#     r.time = 0
#     # TODO: this should reset registers
# end

# TODO: pushinput takes a lot of time, do we need it for ffnn?  
# Yes, we don't know if the next move is forw or back.