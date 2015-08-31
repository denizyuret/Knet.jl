# TODO: figure out the recursive layer definitions.  whether to append or list.

type RNN; net; inputs; freads; breads; x; h; dh;
    function RNN(a...)
        r = new()
        n = length(a)
        r.net = map(ai->(isa(ai,Tuple) ? ai[1] : ai), a)
        r.inputs = [ Int[] for i=1:n ] # inputs[j]={i: h[j,t] needs h[i,t']}
        r.freads = [ Int[] for i=1:n ] # freads[i]={j: h[j,t] needs h[i,t']}
        r.breads = [ Int[] for i=1:n ] # breads[i]={j:dy[j,t] needs h[i,t']} 
        for j=1:n
            inputs = isa(a[j],Tuple) ? a[j][2:end] : ((j-ninputs(a[j])):(j-1))
            ninputs(r.net[j]) == length(inputs) || error("Wrong number of inputs")
            overwrites(r.net[j]) && in(0, inputs) && error("$j:$(typeof(r.net[j])) overwrites RNN input")
            push!(r.inputs[j], inputs...)
            for i in r.inputs[j]
                (0 <= i <= n) || error("Bad input index")
                (i > 0) && push!(r.freads[i], j)
            end
          end
        for i=1:n
            for j in r.freads[i]
                back_reads_x(r.net[j]) && push!(r.breads[i], j)
            end
            back_reads_y(r.net[i]) && push!(r.breads[i], i)
        end
        return r
    end            
end

function forw(r::RNN, x; o...)
    (N,T) = (length(r.net), length(x))
    inithidden(r, :h, N, T)
    r.x = x
    for t = 1:T
        for n = 1:N
            h = forw(r.net[n], rnninput(r,n,t)...; o...)
            # see if we need to preserve the output for n,t+1:
            if ((n < N) &&                      # preserve the last layer
                (length(r.freads[n]) <= 1) &&	# preserve if multiple forw reads
                (length(r.breads[n]) == 0))     # preserve if any back reads
                r.h[n,t] = h
            elseif r.h[n,t] == nothing          # otherwise create a copy per time step
                r.h[n,t] = copy(h)
            else
                resize!(r.h[n,t], size(h))
                copy!(r.h[n,t], h)
            end
        end
    end
    return vec(r.h[N,:])
end

# TODO: the semantics of forw should be simpler: each unit reads the values of its inputs as they are at that point in time,
# whether they are ahead in the network or behind.

function rnninput(r::RNN, n, t)
    map(r.inputs[n]) do i       # return the input matrix or tuple of input matrices to produce r.h[n,t]
        i == 0 ? r.x[t] :       # index=0 is used to indicate input x
        i < n  ? r.h[i,t] :     # indices < n are outputs from the current time slice
        t > 1  ? r.h[i,t-1] :   # indices >= n are from the previous time slice
        nothing                 # nothing represents zero matrix from t=0
    end
end

function inithidden(r::RNN, x, N, T)
    if !isdefined(r,x)
        r.(x) = fill!(Array(Any, N, T), nothing)
    elseif size(r.(x), 1) != N
        error("Size mismatch")
    elseif size(r.(x), 2) != T
        isa(r.(x), SubArray) && (r.(x) = r.(x).parent)
        if size(r.(x), 2) > T
            r.(x) = sub(r.(x), 1:N, 1:T)
        elseif size(r.(x), 2) < T
            h = Array(Any, N, T)
            copy!(h, 1, r.(x), 1, length(r.(x)))
            h[length(r.(x))+1:end] = nothing
            r.(x) = h
        end
    end
    @assert size(r.(x)) == (N,T)
    return r.(x)
end

function initdiff(r::RNN)
    for l in r.net
        w = param(l)
        w == nothing && continue
        similar!(w, :diff, w.arr)
        similar!(w, :inc, w.arr)
        fill!(w.diff, 0)
    end
end

function back(r::RNN, dy; o...)                 # dy[1:T] loss gradients for output y
    (N,T) = (length(r.net), length(dy))
    @assert length(r.x) == N
    inithidden(r, :dh, N, 1)                    # r.dh[1:N] are loss gradients for hidden r.h[1:N,t]
    initdiff(r)                                 # initialize weight gradients dw to zero
    for t = T:-1:1
        r.dh[N] = dy[t]                         # y[t]=r.h[N,t] so dy[t]=r.dh[N]
        for n = N:-1:1
            r.dh[n] == nothing && continue      # 'nothing' represents 0 loss gradient
            ni = ninputs(r.net[n])
            x = rnninput(r,n,t)
            ni==1 && (x=x[1])                   # handle single input vs tuple input layers
            dx = back(r.net[n], r.dh[n]; x=x, y=r.h[n,t], incr=true) # incr=true increments dw 
            ni==1 && (dx=(dx,))                 # handle single input vs tuple input layers
            for j=1:ni                          # set r.dh using dx for each input of n
                i=r.inputs[n][j]                # i is the j'th input of n
                if (r.dh[i] == nothing)         # dh[i] should be updated using dx[j]
                    r.dh[i] = dx[j]
                elseif length(r.freads[i]) == 1
                    r.dh[i] = dx[j]
                else
                    axpy!(1, dx[j], r.dh[i])    # increment dh[i] if i has more than one output
                end
            end
            if length(r.freads[n]) > 1
                fill!(r.dh[n], 0)               # if n has multiple outputs, reset dh[n]
            end
        end
    end
end

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

# figure out when no back needed
## if the previous layer does not need it (rand, input) do not propagate. TODO: how to detect?

# OK: Initialize w the first time?
# OK: back needs x, y, and incr options.
# OK: get rid of forw.y and back.dx parameters

# OK: dy should be like x, only t component, and refer to the output of net[N,t]
# OK: if dy[t] is nothing do not propagate any error.
# OK: we dont want to alloc dy per time step, we want one dy per layer, let layers manage them.

### DEAD CODE
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

:ok
