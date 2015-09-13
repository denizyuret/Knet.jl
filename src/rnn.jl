# TODO
# + try the adding problem with irnn
# + gradient analysis with adding, should fail
# + add incremental update of dy and confirm gradient testing is now ok
# + let forw and back write the input to their own registers
# x ops with forw=nothing should accept back=nothing input
# + minibatching
# x rethink the 'nothing' optimization, removed from dif but not out?
# + return value for back, returndx option: no need, return if dx specified.
# - implement train/predict and try ffnn mnist experiments: how do we treat sequences and minibatches?
# - performance: figure out when no back needed, no returndx needed
# - testing: add a testnet.jl that constructs random nets and does gradient testing: testlayers could not find the pool bug
# - add gradient clipping
# - performance: do better register optimization
# - if an op is using an external array, it should not store it.
# - if we take xi/yi as a parameter for back maybe the net would not have to remember it?

type RNN; op; inputs; ninputs; save; multi; out; out0; dif; dif0; dif1; stack; sp; dbg;
    function RNN(a...; o...)
        r = new()
        initop(r, a...)
        initinputs(r, a...)
        @assert length(r.op)==length(r.inputs)
        initninputs(r)
        initsave(r)
        initmulti(r)
        initout(r)
        initdif(r)
        initstack(r)
        r.dbg = false
        setparam!(r; o...)
        return r
    end
end

ninputs(r::RNN)=r.ninputs
nops(r::RNN)=length(r.op)
op(r::RNN,n)=r.op[n]
setparam!(r::RNN; o...)=(for l in r.op; setparam!(l; o...); end; r)
update(r::RNN; o...)=(for l in r.op; update(l; o...); end; r)
loss(r::RNN,dy)=(dy==nothing ? 0 : loss(r.op[nops(r)], dy; y=convert(typeof(dy), r.out[nops(r)])))
get1(x)=(length(x)==1?x[1]:x)

function forw(r::RNN, inputs...; train=false, y=nothing, a...)
    length(inputs) == ninputs(r) || error("Wrong number of inputs")
    for i = 1:ninputs(r)
        n = i+nops(r)                           # input[i] goes into out[i+nops(r)]
        eltype(inputs[i]) == eltype(r.out0[n]) || error("Element type mismatch")
        train && r.save[n] && push(r,n)         # t:140 save old input if necessary
        r.out[n] = copy!(r.out0[n], inputs[i]) 	# ; dbg(r,:out,n) # t:98 inputs can be any type of array, this will copy it to gpu or wherever
    end
    for n = 1:nops(r)
        train && r.save[n] && push(r,n)         # t:327
        r.out[n] = forw(r.op[n], r.out[r.inputs[n]]...; y=r.out0[n], a...)     # ;dbg(r,:out,n) # t:2300
    end
    y != nothing && copy!(y, r.out[nops(r)])
    return y
end

function forw(r::RNN, x::Vector; y=nothing, a...)
    init(r, x)
    for i=1:length(x)
        yi = (y == nothing ? nothing : y[i])
        forw(r, x[i]; y=yi, a...)
    end
    return y
end

forw{T<:Number}(r::RNN,  x::Vector{T}; a...)=forw(r, reshape(x,  length(x),  1); a...)
back{T<:Number}(r::RNN, dy::Vector{T}; a...)=back(r, reshape(dy, length(dy), 1); a...)

function back(r::RNN, dy::Vector; dx=nothing, a...)
    for i=length(dy):-1:1
        dxi = (dx == nothing ? nothing : dx[i])
        back(r, dy[i]; dx=dxi, a...)
    end
end

function back(r::RNN, dy; dx=nothing, a...)
    dx == nothing || length(dx) == ninputs(r) || error("Wrong number of inputs")
    n = nops(r)
    if dy == nothing
        r.multi[n] || (r.dif[n] = nothing)
    elseif eltype(dy) != eltype(r.dif0[n])
        error("Element type mismatch")
    elseif r.multi[n]
        copy!(r.dif1[n], dy)
        r.dif[n] = axpy!(1,r.dif1[n],r.dif0[n])
    else
        r.dif[n] = copy!(r.dif0[n], dy)
    end										; dbg(r,:dif,n) 
    for n = nops(r):-1:1
        if r.dif[n] == nothing
            for i in r.inputs[n]
                r.multi[i] || (r.dif[i] = nothing)
            end
        else
            dx = Any[]
            for i in r.inputs[n]
                push!(dx, r.multi[i] ? r.dif1[i] : r.dif0[i])
            end
            back(r.op[n], r.dif[n]; incr=true, x=get1(r.out[r.inputs[n]]), y=r.out[n], dx=get1(dx), a...) # t:2164
            for i in r.inputs[n]
                r.multi[i] && axpy!(1, r.dif1[i], r.dif0[i])            ; r.multi[i]&&dbg(r,:dif1,i)
                r.dif[i] = r.dif0[i]                                    ; dbg(r,:dif,i)
            end
            r.multi[n] && fill!(r.dif[n],0)                           ; r.multi[n]&&dbg(r,:dif,n) # t:157
        end
        r.save[n] && pop(r,n)                                    ; r.save[n]&&dbg(r,:out,n)
    end
    for i = ninputs(r):-1:1
        n = i+nops(r)
        r.save[n] && pop(r,n)                                    ; r.save[n] && dbg(r,:out,n)
        dx == nothing || copy!(dx[i], r.dif[n])
    end
    return dx
end

function push(r::RNN,n::Int)
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

function pop(r::RNN,n::Int)
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

# r.multi[n] is true if out[n] has fanout > 1.
# In which case its dif should be incrementally updated.

function initmulti(r::RNN)
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

function initout(r::RNN)
    nout = nops(r)+ninputs(r)
    index = zeros(Int,nops(r))          # index==0 represents need for new register
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

function initdif(r::RNN)
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

function initstack(r::RNN)
    r.stack = Any[]
    r.sp = 0
end

# This is run by train/predict before the start of a new sequence of forw's and back's
# inputs should be the input that will be fed to forw
# subsequent inputs to forw should have the same size, shape

function init(r::RNN, inputs...)
    r.sp == 0 || error("Stack corruption")
    length(inputs) == ninputs(r) || error("Wrong number of inputs")
    fill!(r.out,nothing)
    for i = 1:ninputs(r)
        n = i+nops(r)
        r.out[n] = initarray(r.out0, n, inputs[i])
    end
    while findfirst(r.out,nothing) > 0
        nalloc = 0
        for n = 1:nops(r)
            r.out[n] == nothing || continue
            s = ysize(r.op[n], r.out[r.inputs[n]]...)
            s == nothing && continue        # may happen with recurrent connections
            r.out[n] = initarray(r.out0, n, r.out[r.inputs[n]][1], s)
            p = param(r.op[n])
            p != nothing && isempty(p) && forw(r.op[n], r.out[r.inputs[n]]...; y=r.out0[n]) # initializes w
            nalloc += 1
        end
        nalloc == 0 && error("Cannot determine size of array")
    end
    for n=1:length(r.dif0)
        initarray(r.dif0, n, r.out[n])
        if r.multi[n]
            fill!(r.dif0[n], 0)
            initarray(r.dif1, n, r.out[n])
        end
    end
    for l in r.op
        w = param(l)
        w == nothing && continue
        similar!(w, :diff, w.arr)
        similar!(w, :inc, w.arr)
        fill!(w.diff, 0)
    end
    fill!(r.out,nothing)
    fill!(r.dif,nothing)
end

# The input x::Vector can be x[i][t][d...,b] or x[t][d...,b]
# where i:instance, t:time, d:dims, b:batch
# We only want to init if x[t][d...,b]
# In that case x[1] will be a numeric array 
# or a tuple of numeric arrays (to support multiple inputs)

function init(r::RNN, x::Vector)
    if isempty(x) || x[1] == nothing
        error("Got nothing as input")
    elseif isa(x[1], Tuple)
        init(r, x[1]...)
    elseif isbits(eltype(x[1]))
        init(r, x[1])
    end
end

function initarray(a, i, x, dims=size(x))
    if isempty(a[i])
        oldai = a[i]
        at = (gpu()?CudaArray:Array)
        xt = eltype(x)
        a[i] = (issparse(x) ? KUsparse(at,xt,dims) : KUdense(at, xt, dims))
        for j=1:length(a); a[j]===oldai && (a[j]=a[i]); end # preserve array sharing
    elseif eltype(a[i]) != eltype(x)
        error("Element type mismatch")
    elseif size(a[i]) != dims
        warn("Resizing $(size(a[i]))->$dims")
        resize!(a[i], dims)
    end
    return a[i]
end


ptr16(x)=hex(x==nothing ? 0 : hash(pointer(x)) % 0xffff, 4)
ptr8(x)=hex(x==nothing ? 0 : hash(pointer(x)) % 0xff, 2)
idx1(x)=(x==nothing ? -1 : atype(x)==CudaArray ? to_host(x)[1] : atype(x)==Array ? x[1] : error("$(typeof(x))"))

function dbg(r,f,n)
    r.dbg || return
    a = r.(f)[n]
    print("\n==> $f[$n]($(ptr16(a))) stack:")
    println(map(ptr16, r.stack[1:r.sp]))
    a != nothing && display(convert(Array,a))
    if n <= nops(r) && param(r.op[n]) != nothing
        p = param(r.op[n])
        println("\nw[$n]($(ptr16(p.arr)))")
        display(convert(Array,p.arr))
        if isdefined(p,:diff)
            println("\ndw[$n]($(ptr16(p.diff)))")
            display(convert(Array,p.diff))
        end
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

# # DONE: this should also look at back needs when allocating

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
#         # DONE: we only need the counts not the elements for freads and breads
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





# function forwinput(r::RNN, x, n)
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

# # r.reg[i] is the i'th register of the Net
# # Each register points to an array allocated elsewhere 
# # or is set to 'nothing' representing the zero matrix
# # These registers are used for network inputs, op outputs, op gradients.
# # initreg(r) just creates an empty array of registers r.reg
# # other init functions below add registers as needed to r.reg

# function initreg(r::RNN)
#     r.reg = Any[]
# end

# function initbuf(r::RNN)
#     r.buf = copy(r.reg) # array of nothings, initialized at initforw
# end

# function initregi(r::RNN, i::Int, y, dims=size(y))
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
# op(r::RNN,n::Int)=r.op[n]

# gety(r::RNN,n::Int)=r.reg[r.out[n]]
# getdy(r::RNN,n::Int)=r.reg[r.dif[n]]
# getx(r::RNN,n::Int)=r.reg[r.out[r.inputs[n]]]
# getdx(r::RNN,n::Int)=r.reg[r.dif[r.inputs[n]]]

# getbuf(r::RNN,i::Int)=(r.reg[i]!=nothing?r.reg[i]:r.buf[i])
# getybuf(r::RNN,n::Int)=getbuf(r,r.out[n])
# getdybuf(r::RNN,n::Int)=getbuf(r,r.dif[n])
# getxbuf(r::RNN,n::Int)=map(i->getybuf(r,i),r.inputs[n])
# getdxbuf(r::RNN,n::Int)=map(i->getdybuf(r,i),r.inputs[n])

# get1x(r::RNN,n::Int)=get1(getx(r,n))
# get1dxbuf(r::RNN,n::Int)=get1(getdxbuf(r,n))

# sety(r::RNN,n::Int,x)=(r.reg[r.out[n]]=x)
# setdy(r::RNN,n::Int,x)=(r.reg[r.dif[n]]=x)
# setinput(r::RNN,n::Int,x)=(r.reg[r.out[n+nops(r)]]=x)
# getinput(r::RNN,n::Int)=r.reg[r.out[n+nops(r)]]

# pushy(r::RNN,n::Int)=(r.save[n] && pushreg(r,r.out[n]))
# popy(r::RNN,n::Int)=(r.save[n] ? popreg(r,r.out[n]) : r.reg[r.out[n]])
# pushinput(r::RNN,n::Int)=(r.save[n+nops(r)] && pushreg(r,r.out[n+nops(r)]))
# popinput(r::RNN,n::Int)=(r.save[n+nops(r)] && popreg(r,r.out[n+nops(r)]))

# function pushreg(r::RNN,i::Int)
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

# function popreg(r::RNN,i::Int)
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

# DONE: incremental updates
 # DONE: forw: do we really need inputs to be tuple here? YES, have the option of multi-input nets like multi-input ops.
# DONE: forw: (minor) this ends up using a lot of nothing pushes first minibatch, that's ok, 'nothing' is a legit value.

