"""
initback initializes the fields used by Net.back:
- dif, dif0, tmp: allocated or size checked.
- toincr: depends on seq.
- toback: depends on which dx args specified.
- tosave: read-only, used for popping only if seq.
"""
function initback(r::Net, ygold, loss; getdx=false, seq=false, a...)
    @assert ygold == nothing || issimilar2(ygold, r.out0[end])
    set_toback(r, getdx)
    set_toincr(r, seq)
    set_sparse(r)
    for n=length(r.op):-1:1
        r.toback[n] || continue # we mix initback and initback0 here because if getdx or seq changes we may need to alloc new arrays.
        if isassigned(r.dif0,n) # TODO: implement batch size change
            @assert (issimilar2(r.dif0[n], r.out0[n]) ) #TODO: && issparse(r.dif0[n])==r.sparse[n])
        else
            r.dif0[n] = finddif(r, n)
            r.dif[n] = nothing
        end
        if r.toincr[n]
            if isassigned(r.tmp, n)
                @assert (issimilar2(r.tmp[n], r.out0[n]) && issparse(r.tmp[n])==r.sparse[n])
            else
                r.tmp[n] = findtmp(r, n)
                fill!(r.dif0[n], 0)
            end
        end
    end
end

"""
set_toback(r::Net) sets r.toback[n] which is true if dif[n] should be
calculated for op[n] during back calculation.  This is only needed if 
op[n] is a par node or a par node descendent.  Or if the caller asked
for dx for network inputs, those and their descendents.
"""
function set_toback(r::Net, getdx; a...)
    fill!(r.toback, false)
    N = length(r.op)
    lastinput = 0
    for n=1:N
        isa(r.op[n], Par) && (r.toback[n] = true)
        isa(r.op[n], Input) && getdx && (r.toback[n] = true)
    end
    nback = sum(r.toback)
    while true
        for n=1:N
            r.toback[n] && continue
            for i in r.inputs[n]
                if r.toback[i]
                    r.toback[n] = true
                    break
                end
            end
        end
        nb = sum(r.toback)
        nb == nback ? break : nback = nb
    end
end


"""
set_toincr(r::Net) sets r.toincr[n] which is true if dif[n] should be
incrementally updated.  This is necessary if op[n] has multiple
outputs, or it is a Par and we are processing a sequence.  This is the
one place we can't seem to get rid of the seq flag.
"""
function set_toincr(r::Net, seq)
    fill!(r.toincr, false)
    for n=1:length(r.op)
        length(r.outputs[n]) > 1 && (r.toincr[n] = true)
        seq && isa(r.op[n], Par) && (r.toincr[n] = true)
    end
end


function finddif(r::Net, n)
    dif0 = nothing
    if !r.toincr[n]
        @assert length(r.outputs[n]) == 1 # otherwise toincr would be true
        o = r.outputs[n][1]     # first try overwriting the output dif
        if (o > n
            && !r.toincr[o]
            && overwrites(r.op[o])
            && isassigned(r.dif0, o)
            && size(r.dif0[o]) == size(r.out0[n]))
            dif0 = r.dif0[o]
        end
    end
    if dif0 == nothing
        et = eltype(r.out0[n])
        sz = size(r.out0[n])
        dif0 = (r.sparse[n] && !r.toincr[n] && gpu()  ? CudaSparseMatrixCSRU(et, sz...) :
                r.sparse[n] && !r.toincr[n] && !gpu() ? spzeros(et, sz...) :
                ### Uncomment this if you want sparse incremental dw:
                ### Similar speed, less memory, however cannot compute vecnorm.
                # r.sparse[n] && r.toincr[n] && gpu()   ? ArrayAccumulator(et, sz) :
                # r.sparse[n] && r.toincr[n] && !gpu()  ? ArrayAccumulator(et, sz) :
                gpu() ? CudaArray(et, sz) : Array(et, sz))
    end
    return dif0
end

function findtmp(r::Net, n)
    tmp = nothing
    for i=n+1:length(r.op)
        if (isassigned(r.tmp, i) &&
            size(r.tmp[i]) == size(r.dif0[n]) &&
            issparse(r.tmp[i]) == r.sparse[n])
            tmp = r.tmp[i]
            break
        end
    end
    if tmp == nothing
        et = eltype(r.out0[n])
        sz = size(r.out0[n])
        tmp = (gpu() && r.sparse[n] ? CudaSparseMatrixCSRU(et, sz...) :
               !gpu() && r.sparse[n] ? spzeros(et, sz...) :
               gpu() ? CudaArray(et, sz) : 
               Array(et, sz))
    end
    return tmp
end

# The only sparse matrices are the dw and iw for w that dot sparse inputs.
function set_sparse(r::Net)
    fill!(r.sparse, false)
    for i=1:length(r.op)
        if isa(r.op[i], Input) && issparse(r.out0[i])
            for o in r.outputs[i]
                if isa(r.op[o], Dot)
                    w = r.inputs[o][1]
                    @assert isa(r.op[w], Par)
                    r.sparse[w] = true
                end
            end
        end
    end
end


### DEAD CODE

# # instead of changing the ops, record the eltype in net and use it when generating stuff.
#     # for o in r.op
#     #     t = eltype(o)
#     #     if t == nothing
#     #         continue
#     #     elseif it == nothing
#     #         it = t
#     #     elseif t != it
#     #         seteltype!(o, it)
#     #     end
#     # end

# #####

# function netinit(r::Net)
#     initout(r, inputs...)
#     initdif(r, inputs...)
#     inittmp(r, inputs...)
# end

# """

# initout(r::Net, inputs...) initializes r.out[n] which points to the
# array holding the output of op[n].  r.out0[n] holds the actual array,
# allowing r.out[n] point to stack elements when necessary during back
# calculation.  We optimize memory use by sharing arrays when we can.

# """
# function initout(r::Net)
#     N = length(r.op)
#     index = zeros(Int,N)          			# index==0 represents need for new register
#     for n=1:N                    
#         overwrites(r.op[n]) || continue                 # see if we can reuse our input for output
#         r.tosave[n] && continue                         # a saved register should only be written by op[n]
#         i = r.inputs[n][1]                              # see if we can overwrite the first input
#         r.tosave[i] && continue                         # do not overwrite if you are going to save for back
#         ow = true; k = n                                # TODO: checkout other inputs if this one fails
#         while true                                      # see if anybody else uses i before its next update
#             k = mod1(k+1, N)                            # TODO: This is suboptimal, gives 13 regs for LSTM
#             in(i, r.inputs[k]) && (ow = false; break)   # Should look for existing regs no longer used
#             k == i && break                             # this is nontrivial because the sizes are unknown
#             k == N && i > N && break                    # but can possibly be inferred
#         end                                             # fencepost check: N==1:OK, i==n:OK, i>N:OK 
#         ow && (index[n]=i)
#     end
#     r.out0 = cell(N)
#     for n=1:N                                           # index==0 ops get their own registers
#         index[n]==0 || continue 
#         r.out0[n] = Any[]
#     end
#     for n=1:N                                             # other ops will overwrite existing registers
#         index[n]==0 && continue
#         k = index[n]
#         while index[k]!=0; k=index[k]; end
#         r.out0[n] = r.out0[k]
#     end
#     r.out = fill!(cell(N), nothing)                       # use nothing to represent identity element: 0 for add, 1 for mul
# end

# """
# initdif(r::Net) initializes r.dif[n], which is the loss gradient of the last output of op[n].
# r.dif0[n] holds the actual array, r.dif[n] can point to this or be 'nothing' representing the zero matrix.
# """
# function initdif(r::Net)
#     # TODO: this is not optimal for add: no need for two copies of dy when neither input is overwriting.
#     # TODO: tmp register to replace dif1 and serve other needs

#     index = zeros(Int,nops(r))                                  # index==0 represents need for new register
#     for n=1:nops(r)                                             # find out if back(op[n]) can overwrite dif[n]
#         overwrites(r.op[n]) || continue                         # overwrites means potentially both forw x<-y and back dy<-dx
#         r.toincr[n] && continue                                  # don't share dif[n] with multi-output
#         n == nops(r) && continue                                # don't share last output, dy may come in sparse
#         for i in r.inputs[n]
#             r.toincr[i] && continue                              # don't share dif[n] with multi-output
#             index[i]==0 || error("should not happen")
#             index[i]=n; break                                   # op[n] will overwrite dif[n] to get dif[i]
#         end
#     end
#     r.dif0 = cell(nops(r))
#     for n=1:nops(r)
#         index[n]==0 || continue
#         r.dif0[n] = Any[]
#     end
#     for n=1:nops(r)
#         index[n]==0 && continue
#         k = index[n]
#         while index[k]!=0; k=index[k]; end
#         r.dif0[n] = r.dif0[k]
#     end
#     r.dif = fill!(cell(nops(r)), nothing)
# end

# """
# inittmp(r::Net) initializes r.tmp[n] which is an extra array for:
# (1) incremental updates if r.toincr[n].
# (2) for forw/back ops that need extra space (TODO)
# """
# function inittmp(r::Net)
#     # TODO: this is inefficient, could use a single tmp for each size.
#     r.tmp = cell(nops(r))
#     for n=1:length(r.tmp)
#         r.tmp[n] = r.toincr[n] ? Any[] : nothing
#     end
# end

# ### DEAD CODE:


#     # initop(r, a...)
#     # initinputs(r, a...)
#     # @assert length(r.op)==length(r.inputs)
#     # initninputs(r)
#     # initparams(r)
# # function initstack(r::Net)
# #     r.stack = Any[]
# #     r.sp = 0
# # end

# #    initstack(r)

# # r.op[n] is the n'th operation in the net
# # The user specifies the operations in the MLP constructor arguments
# # If an argument is another MLP, its operations are spliced in

# # function initop(r::Net, a...)
# #     r.op = Op[]
# #     for ai in a
# #         isa(ai,Tuple) && (ai=ai[1])
# #         isa(ai,Op) ?  push!(r.op, ai) :
# #         isa(ai,Net)   ?  append!(r.op, ai.op) :
# #         error("Bad op: $ai")
# #     end
# # end

# # r.inputs[n] is an array of k indices for the inputs of the n'th op.
# # k is typically 1 but can be 2 for Add2 and Mul2 operations
# # index i<=nops(r) indicates output of r.op[i], i>nops(r) indicates network inputs
# # By default an op takes the results of the previous k op outputs
# # (or network inputs for initial ops)
# # The user can override this by providing a tuple argument for an op
# # The first element of the tuple is the op, the rest are user indices for inputs
# # userindex j>0 indicates output of userarg[j], j<=0 indicates network input 1-j.

# # function initinputs(r::Net, a...)
# #     newindex = Array(Int, length(a))
# #     lastindex = 0
# #     for i=1:length(a)
# #         ai = isa(a[i],Tuple) ? a[i][1] : a[i]
# #         lastindex += (isa(ai,Op) ? 1 : length(ai.op))
# #         newindex[i] = lastindex
# #     end
# #     r.inputs = Any[]
# #     for i=1:length(a)
# #         ai = isa(a[i],Tuple) ? a[i][1] : a[i]
# #         bi = isa(a[i],Tuple) ? a[i][2:end] : ((i-ninputs(ai)):(i-1))
# #         length(bi) == ninputs(ai) || error("Wrong number of inputs for $i:$(typeof(ai))")
# #         if isa(ai, Op)
# #             push!(r.inputs, map(j->(j>0 ? newindex[j] : nops(r)+1-j), Int[bi...]))
# #         else
# #             j0 = length(r.inputs)
# #             for aii in ai.inputs
# #                 push!(r.inputs, map(j->(j<=nops(ai) ? j+j0 : (j=bi[j-nops(ai)]; j>0 ? newindex[j] : nops(r)+1-j)), aii))
# #             end
# #         end
# #     end
# # end

# # r.ninputs is the number of inputs the whole MLP expects
# # indices i>nops(r) in r.inputs refer to network inputs

# # function initninputs(r::Net)
# #     n = 0
# #     for ai in r.inputs
# #         for aij in ai
# #             aij - nops(r) > n && (n = aij - nops(r))
# #         end
# #     end
# #     r.ninputs = n
# # end

# # r.params points to all op parameters

# # function initparams(r::Net)
# #     r.params = Any[]
# #     for o in r.op
# #         if isa(o,par) && isupdated(o)
# #             push!(r.params, o)
# #         end
# #     end
# # end

# # isupdated(o)=true # TODO: should be false for rand and const

#     #     @assert !haskey(dict, target)			# each target has to be unique
#     #     for x in args
#     #         @assert haskey(dict, x)                     # 
#     #     end
#     #     if func == :input && inputs != nothing          # subroutines take their input registers from caller
#     #         @assert length(inputs) >= nextinput
#     #         dict[target] = inputs[nextinput]
#     #         nextinput += 1
#     #     end
#     #     op = eval(Expr(:call, func, pars...))
#     # end
#     # 

# # initforw(r::Net,x...) is called at the beginning of a sequence or
# # before processing a stand-alone item.  It is not called between
# # elements of a sequence.  It allocates and/or resizes r.out0.

# # function initforw2(r::Net, inputs...; keepstate=false, a...)
# #     r.dbg && display((:initforw0,keepstate,vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
# #     r.sp == 0 || error("Stack corruption")
# #     length(inputs) == ninputs(r) || error("Wrong number of inputs")
# #     out = fill!(cell(length(r.out0)), nothing)          # TODO: (minor) can we get rid of alloc
# #     lastinput = 0
# #     while findfirst(out,nothing) > 0                            # allow multiple passes for size inference
# #         nnothing = count(x->(x==nothing), out)
# #         for n = 1:nops(r)
# #             if out[n] != nothing
# #                 continue
# #             elseif isa(r.op[n], Input)
# #                 out[n] = initarray(r.out0, n, inputs[lastinput += 1])
# #             elseif isempty(r.inputs[n])                         # par, rnd, con
# #                 s = psize(r, n)                                 # TODO: write psize inference
# #                 s == nothing && continue                        # may happen first pass
# #                 out[n] = initarray(r.out0, n, inputs[1], s; dense=true)
# #             else
# #                 s = ysize(r.op[n], out[r.inputs[n]]...)
# #                 s == nothing && continue # may happen with recurrent connections
# #                 out[n] = initarray(r.out0, n, out[r.inputs[n]][1], s; dense=true)
# #             end
# #         end
# #         nnothing == count(x->(x==nothing), out) && error("Cannot determine size of array $(findfirst(out,nothing))")
# #     end
# #     # We recover or reset r.out:
# #     keepstate ? copy!(r.out, r.out0) : fill!(r.out, nothing) # TODO: r.out[i] = arr(r.out0[i]) elsewhere..
# #     r.dbg && display((:initforw1,keepstate,vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
# # end

# # function initarray(a, i, x, dims=size(x); dense=false)
# #     (display((:initarray0,summary(a[i]),i,summary(x),dims,dense,summary(a)));println())
# #     if isempty(a[i])
# #         oldai = a[i]
# #         if !dense && issparse(x)
# #             a[i] = spzeros(eltype(x), dims...)
# #             gpu() && (a[i] = CudaSparseMatrixCSC(a[i]))
# #         else
# #             a[i] = fill!(KUdense(gpu()?CudaArray:Array, eltype(x), dims), 0)
# #         end
# #         for j=1:length(a); a[j]===oldai && (a[j]=a[i]); end # preserve array sharing
# #     elseif eltype(a[i]) != eltype(x)
# #         error("Element type mismatch")
# #     elseif size(a[i]) != dims
# #         warn("Resizing $(size(a[i]))->$dims")
# #         fill!(resize!(a[i], dims), 0) # TODO: this does not work for sparse
# #     end
# #     (display((:initarray1,summary(a[i]),i,summary(x),dims,dense,summary(a)));println())
# #     return a[i]
# # end

#             # elseif in(nothing, dims[r.inputs[n]])
#             #     continue
#             # else
#             # end

#             # if isempty(r.inputs[n])                     # happens with par, rnd, con, input
#             #     elseif prod(size(r.op[n])) > 0
#             #         dims[n] = size(r.op[n])
#             #     end
#             # end
#             # if isa(op, Input)
#             #     dims[n] = size(inputs[lastinput += 1])
#             # elseif isa(op, Par) && prod(op.dims) > 0
#             #     dims[n] = op.dims
#             # elseif isa(op, Dot)
#             #     (i1,i2) = r.inputs[n]
#             #     (d1,d2) = dims[i1,i2]
#             #     if d1 == nothing || d2 == nothing
#             #         continue
#             #     end
#             #     m1 = prod(d1[1:end-1])                   # treat [d...,dn] as [prod(d...), dn] matrix
#             #     n1 = d1[end]
#             #     m2 = prod(d2[1:end-1])
#             #     n2 = d2[end]
#             #     if m1 == 0 || n2 == 0
#             #         continue
#             #     end
#             #     if n1 == m2 == 0
#             #         continue
#             #     elseif n1 == 0
#             #         setdims!(r.op[i1], (d1[1:end-1]..., m2))
#             #     elseif m2 == 0
#             #         length(d2) == 2 || error("Cannot infer size of tensor")
#             #         setdims!(r.op[i2], (n1, d2[2]))
#             #     elseif n1 == m2
#             #         dims[n] = (m1,n2)
#             #     else
#             #         error("Dot size mismatch")
#             #     end
#             # end
#             # # input, par, dot, add, mul, actf, loss, conv, pool

# function initback0(r::Net, dy; a...)
#     N = length(r.op)
#     firstwrite = zeros(Int, N)                          # dif[n] is read at t=n, find out when it is first written.
#     # TODO: dif[N] is special!
#     for n=N:-1:1
#         r.toback[n] || continue
#         k = n
#         while true
#             in(n, r.inputs[k]) && (firstwrite[n] = k; break)
#             k = mod1(k-1, N)
#         end
#     end
#     return firstwrite
#     for n=N:-1:1
#         nsparse = false # TODO
#         r.toback[n] || continue
#         r.dif[n] = finddif(r, n, nsparse)
#         r.toincr[n] && (r.tmp[n] = findtmp(r, n, nsparse))
#     end
# end

# function finddif(r::Net, n, nsparse)
#     isa(r.op[n], Par) && return                         # par difs should persist across iterations
#     N = length(r.op)
#     k = n; nw1 = 0                                      # dif[n] is read at t=n, find out when it is first written.
#     while true
#         in(n, r.inputs[k]) && (nw1 = k; break)          # it could potentially be written at t=n right after reading
#         k = mod1(k-1, N)
#     end
#     @assert nw1 > 0                                     # dif[n] will be busy from nw1 to n going back cyclical
#     free = nothing                                      # we need dif[i] that is free during this period
#     for i = n+1:N                                       # note that dif[i] may already be shared to some dif[j]
#         isa(r.op[i], Par) && return
#         size(r.dif[i]) == size(r.out0[n]) || continue
#         issparse(r.dif[i]) == nsparse || continue
        
#     end
#     return free
# end

# initback(r::Net, dy) called at the beginning of a sequence or a
# stand-alone item, never between elements of a sequence.

# function initback1(r::Net, dy; seq=false, a...)
#     fill!(r.dif, nothing)                           # why? (TODO)
#     for n=1:length(r.dif0)
#         y = (n==nops(r) && dy!=nothing ? dy : r.out0[n])
#         initarray(r.dif0, n, y; dense=true) # x and dw may be sparse, dx and w always dense
#         if r.toincr[n]
#             initarray(r.tmp, n, r.dif0[n])
#         end
#     end
#     for n=1:length(r.dif0)
#         r.toincr[n] && fill!(r.dif0[n], 0)           # zeroed by back at every item in sequence
#     end
#     if seq
#         for w in params(r)       # zeroed only once at the beginning of the sequence
#             isdefined(w,:diff) && isdefined(w,:inc) && fill!(w.diff,0)
#         end
#     end
# end

# function initback0(r::Net, dy, dx...; a...)
#     N = length(r.op)
#     dxback = falses(N)
#     if !isempty(dx)                                     # TODO: what if this changes after net init?
#         lastinput = 0
#         for n=1:N
#             isa(r.op[n], Input) || continue
#             dx[lastinput += 1] == nothing || (dxback[n] = true)
#         end
#     end
#     for n=N:-1:1
#         nsparse = difsparse(r, dy, n)
#         r.toback[n] || dxback[n] || continue
#         # r.dif0[n] = finddif(r, n, nsparse)  # TODO-OPTIMIZATION
#         r.dif0[n] = newarray(gpu(), nsparse, eltype(r.out0[n]), size(r.out0[n]))
#         if r.toincr[n]    # TODO: what if this changes after init?
#             # r.tmp[n] = findtmp(r, n, nsparse) # TODO-OPTIMIZATION
#             r.tmp[n] = newarray(gpu(), nsparse, eltype(r.out0[n]), size(r.out0[n]))
#         end
#     end
# end

        # else                    # otherwise find one with matching size
        #     for k=o+1:length(r.op)
        #         if (!r.toincr[k]
        #             && isassigned(r.dif0, k)
        #             && size(r.dif0[k]) == size(r.out0[n])
        #             && stype(r.dif0[k]) == st
        #             && r.outputs[k][1] > k)
        #             dif0 = r.dif0[k]
        #             # TODO: However we need to check and see if this has been used between n..o
        #             # Also we don't know whether o > n for sure
        #             # This all needs more thinking
        #             break
        #         end
        #     end
# should be part of reset:
    # fill!(r.dif, nothing)
    # for n=1:length(r.op)
    #     isassigned(r.dif0, n) && r.toincr[n] && fill!(r.dif0[n], 0)
    # end

