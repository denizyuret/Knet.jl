type Stack; st; sp; Reg()=new(Any[],0); end


type RNN; op; inputs; output; buffer; stack; time;
    function RNN(a...)
        r = new()
        r.op = initops(a...)
        r.inputs = initinputs(a...)
        r.output = initoutput(r)
        r.buffer = initbuffer(r)
        r.stack = initstack(r)
        r.time = 0
        return r
    end
end

function forw(r::RNN, x...; predict=false, o...)
    r.time += 1
    allocbuffer(r, x)
    !predict && allocstack(r, x)
    N = length(r.op)
    for n = 1:N
        r.output[n] = forw(r.op[n], forwinput(r,x,n)...; y=r.buffer[n], o...)
        !predict && isassigned(r.stack,n) && copy!(r.stack[n][r.time], r.output[n])
    end
    return r.output[N]
end

function forwinput(r::RNN, x, n)
    map(r.inputs[n]) do i               # return the input matrix or tuple of input matrices to produce r.output[n]
        i <= 0     ? x[1-i] :           # i <= 0 is used to indicate input x[1-i]
        i < n      ? r.output[i] :      # 0 < i < n are outputs from the current time slice
        r.time > 1 ? r.output[i] :	# i >= n are from the previous time slice if r.time > 1
        nothing                         # nothing represents zero matrix from r.time=0
    end
end

function reset(r::RNN)
    r.time = 0
end

function ninputs(net::RNN)
    n = 0
    for ai in net.inputs
        for aij in ai
            # aij=0,-1,... refer to external inputs
            1-aij >= n && (n=1-aij)
        end
    end
    return n
end


### Net initialization

# r.op[n] is the n'th operation in the net
# The user specifies the operations in the Net constructor arguments
# If an argument is another Net, its operations are spliced in

function initops(a...)
    op = Layer[]
    for ai in a
        isa(ai,Tuple) && (ai=ai[1])
        isa(ai,Layer) ?  push!(op, ai) :
        isa(ai,RNN)   ?  append!(op, ai.op) :
        error("Bad op: $ai")
    end
    return op
end

# r.inputs[n] is an array of k indices for the inputs of the n'th operation
# k is typically 1 but can be 2 for Add2 and Mul2 operations
# index i>0 indicates output of r.op[i], i<=0 indicates network inputs
# By default an op takes the results of the previous k op outputs
# The user can override this by providing a tuple argument for an op
# The first element of the tuple is the op, the rest are argument indices
# The user supplied indices refer to the argument position
# Splicing of argument Nets complicate the index calculation a bit
# The n'th operation is not always the n'th argument with splicing

function initinputs(a...)
    newindex = Array(Int, length(a))
    lastindex = 0
    for i=1:length(a)
        ai = isa(a[i],Tuple) ? a[i][1] : a[i]
        lastindex += (isa(ai,Layer) ? 1 : length(ai.op))
        newindex[i] = lastindex
    end
    inputs = Any[]
    for i=1:length(a)
        ai = isa(a[i],Tuple) ? a[i][1] : a[i]
        bi = isa(a[i],Tuple) ? a[i][2:end] : ((i-ninputs(ai)):(i-1))
        length(bi) == ninputs(ai) || error("Wrong number of inputs for $i:$(typeof(ai))")
        if isa(ai, Layer)
            push!(inputs, map(j->(j>0 ? newindex[j] : j), Int[bi...]))
        else
            j0 = length(inputs)
            for aii in ai.inputs
                push!(inputs, map(j->(j>0 ? j+j0 : (j=bi[1-j]; j>0 ? newindex[j] : j)), aii))
            end
        end
    end
    inputs
end

# r.output[n] points to the last output of r.op[n]
# its value could be an array (provided by r.buffer[n]) or 'nothing'
# 'nothing' is used to represent the zero matrix at time=0
# and others that result from them at time>0

function initoutput(r::RNN)
    r.output = fill!(Array(Any,length(r.op)), nothing)
end

# r.buffer[n] is the array pre-allocated for the output of r.op[n]
# r.buffer is initialized in three steps:
# - initbuffer determines the indices for array reuse.
#   i=0 means allocate, i>0 means reuse r.buffer[i]
# - allocbuffer allocates 0 size KUdense arrays
#   this is a separate step because we need input x for type info
# - the actual op resize the buffers as needed.

function initbuffer(r::RNN)
    N = length(r.op)
    index = fill!(Array(Any,N), 0)
    for n=1:N
        if overwrites(r.op[n])
            index[n] = i = r.inputs[n][1] # tentatively prepare to overwrite first input
            k = n  
            while true # see if anybody else uses i before its next update
                k = mod1(k+1, N)
                in(i, r.inputs[k]) && (index[n] = 0; break)
                k == i && break
            end # fencepost check: N==1:OK, i==n:OK
        end
    end
    # info("$(sum(index.==0)) buffers will be allocated.")
    return index
end

function allocbuffer(r::RNN, x)
    isa(r.buffer[1], Int) || return
    N = length(r.op)
    for n=1:N
        if r.buffer[n] == 0
            r.buffer[n] = KUdense(atype(x[1]), eltype(x[1]), (0,))
        end
    end
    for n=1:N
        isa(r.buffer[n], Int) || continue
        k = n
        while isa(r.buffer[k], Int)
            k = r.buffer[k]
        end
        r.buffer[n] = r.buffer[k]
    end
end

# r.stack[n<=N][t] is a copy of the output of r.op[n] at r.time=t
# r.stack[n>N][t] is a copy of the n-N'th Net input at r.time=t
# - where N is the number of op
# - remember Net inputs are indicated by 0,-1,-2 etc. in r.inputs
# We only keep the copies necessary for the back calculation
# - initstack initializes necessary r.stack[n] with empty arrays
#   the other elements in r.stack are left unassigned
# - allocstack makes sure at least r.time arrays are allocated
#   for each assigned r.stack[n]
#   allocstack also copies inputs x as necessary

function initstack(r::RNN)
    N = length(r.op)
    hist = Array(Any,N+ninputs(r))
    for n=1:N
        back_reads_y(r.op[n]) && !isassigned(hist,n) && (hist[n]=Any[])
        if back_reads_x(r.op[n])
            for i in r.inputs[n]
                i <= 0 && (i = N+1-i)
                !isassigned(hist,i) && (hist[i]=Any[])
            end
        end
    end
    return hist
end

function allocstack(r::RNN, x)
    N = length(r.op)
    for n=1:N
        isassigned(r.stack,n) || continue
        length(r.stack[n]) >= r.time && continue
        length(r.stack[n]) == r.time - 1 || error("Stack corruption")
        push!(r.stack[n], KUdense(atype(x[1]), eltype(x[1]), (0,)))
    end
    for i=1:length(x)
        n=N+i
        isassigned(r.stack,n) || continue
        if length(r.stack[n]) < r.time
            length(r.stack[n]) == r.time - 1 || error("Stack corruption")
            push!(r.stack[n], copy(x[i]))
        else
            copy!(r.stack[n][r.time], x[i])
        end
    end
end

### DEAD CODE

# function forw1(r::RNN, x; predict=false, o...)
#     predict ? 
#     forwpredict(r, x; o...) : 
#     forwtrain(r, x; o...)
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

# function initdiff(r::RNN)
#     for l in r.net
#         w = param(l)
#         w == nothing && continue
#         similar!(w, :diff, w.arr)
#         similar!(w, :inc, w.arr)
#         fill!(w.diff, 0)
#     end
# end

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
