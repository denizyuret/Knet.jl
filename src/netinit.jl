# TODO: initback

function initforw(r::Net, inputs...; keepstate=false, o...)
    @assert length(inputs) == ninputs(r)
    @assert r.sp == 0
    isassigned(r.out0, 1) ? 
    initforw1(r, inputs...) :
    initforw0(r, inputs...)
    keepstate || initforw2(r)
    copy!(r.out, r.out0)
end

# first init: infer and alloc
function initforw0(r::Net, inputs...)
    atype = gpu() ? CudaArray : Array
    xtype = infertype(r, inputs...)
    sizes = infersize(r, inputs...)
    for n=1:length(r.op)
        r.out0[n] = allocout0(r, n, atype, xtype, sizes)
    end
    # TODO: figure out sharing and tmp and sparse
end

function allocout0(r::Net, n, atype, xtype, sizes)
    r.tosave[n] && return atype(xtype, sizes[n])        # saved regs and pars should not overwrite or be overwritten
    isa(r.op[n], Par) && return atype(xtype, sizes[n])  # TODO: how about rnd and con?
    free = nothing
    for i = n-1:-1:1                                    # considering overwriting i with n
        r.tosave[i] && continue
        isa(r.op[i], Par) && continue
        size(r.out0[i]) == sizes[n] || continue
        willberead = false                              # is anybody going to read i before it is written again?
        k = n
        while true
            k = mod1(k+1, length(r.op))
            for j in r.inputs[k]
                isassigned(r.out0, j) && r.out0[j] === r.out0[i] && (willberead = true; break)
            end
            willberead && break
            isassigned(r.out0,k) && r.out0[k] === r.out0[i] && break
        end
        !willberead && (free = r.out0[i]; break)
    end
    return (free != nothing ? free : atype(xtype, sizes[n]))
end

function infersize(r::Net, inputs...)
    N = length(r.op)
    dims = fill!(cell(N), nothing)
    lastinput = 0
    notfound = N
    while notfound > 0
        for n=1:N
            if isa(r.op[n], Input)
                dims[n] == nothing && (dims[n] = size(inputs[lastinput += 1]))
            else
                d = infersize(r.op[n], dims[r.inputs[n]]...)
                d == nothing && continue
                dims[n] = d[end]
                dims[r.inputs[n]] = [d[1:end-1]...]
            end
        end
        nf = count(x->(x==nothing || prod(x)==0), dims)
        nf == notfound && error("Cannot infer sizes: $dims")
        notfound = nf
    end
    return dims
end

function infertype(r::Net, inputs...)
    it = nothing
    for i in inputs
        t = eltype(i)
        if it == nothing
            it = t
        elseif t != it
            error("Conflicting input eltypes")
        end
    end
    it == nothing && error("Cannot infer eltype")
    return it
    # TODO: deal with inputless networks:
end

# net already initialized, just check the sizes
function initforw1(r::Net, inputs...)
    N = length(r.op)
    lastinput = 1
    for n=1:N
        isa(r.op[n], Input) || continue
        i = inputs[lastinput += 1]
        o = r.out[n]
        @assert eltype(i) == eltype(o)
        @assert size(i) == size(o)
    end
    # TODO: implement batch size changes
end

# zero out arrays that are read before written
function initforw2(r::Net)
    for n=1:length(r.op)
        r.tozero[n] && fill!(r.out0[n], 0)
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

