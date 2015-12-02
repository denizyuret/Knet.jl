"""
initback initializes the fields used by Net.back:
- dif0, tmp: allocated or size checked.
- reg.incr: depends on seq.
- reg.back: depends on which dx args specified.
- op.save: read-only, used for popping only if seq.
"""
function initback(f::Net, ygold, loss; getdx=false, seq=false, a...)
    # TODO: avoid doing any of this if there is no change in the inputs and flags.
    # This is not true for nce; ygold and f.out0[end] have different dimensions.
    # @assert ygold == nothing || issimilar2(ygold, f.out0[end])
    initgrad(f, getdx)
    initincr(f, seq)
    inittype2(f)
    for r in registers(f)
        if get(r,:grad)
            if !checkarray(r, :dif0, r.diftype, r.eltype, r.size)
                r.dif0 = newarray(r.diftype, r.eltype, r.size)
            end
            if get(r,:incr)
                if !checkarray(r, :dif0, r.diftype, r.eltype, r.size)
                    r.tmp = newarray(r.tmptype, r.eltype, r.size)
                end
            end
        end
    end
end

# """ REWRITE:
# set_toback(f::Net) sets f.toback[n] which is true if dif[n] should be
# calculated for op[n] during back calculation.  This is only needed if 
# op[n] is a par node or a par node descendent.  Or if the caller asked
# for dx for network inputs, those and their descendents.
# """
function initgrad(f::Net, getdx; a...)
    for r in registers(f)
        set!(r,:grad,false)
    end
    lastinput = 0
    for p in instructions(f)
        get(p,:forw) || continue
        if (isa(p.op, Par) ||
            (isa(p.op, Input) &&
             getdx[lastinput+=1]))
            r = output_register(f,p)
            setprop!(r,:grad,true)
        end
    end
    count_back(f::Net)=mapreduce(r->get(r,:grad), +, 0, registers(f))
    nback = count_back(f)
    while true
        for p in instructions(f)
            get(p,:forw) || continue
            r = output_register(f,p)
            get(r,:grad) && continue
            for i in input_registers(f,p)
                if get(i,:grad)
                    set!(r,:grad,true)
                    break
                end
            end
        end
        nb = count_back(f)
        nb == nback ? break : nback = nb
    end
end

# """ REWRITE:
# set_toincr(f::Net) sets f.toincr[n] which is true if dif[n] should be
# incrementally updated.  This is necessary if op[n] has multiple
# outputs, or it is a Par and we are processing a sequence.  This is the
# one place we can't seem to get rid of the seq flag.
# """
function initincr(f::Net, seq)
    for r in registers(f)
        set!(r,:fanout,0)
        set!(r,:incr,false)
    end
    for p in instructions(f)
        get(p,:forw) || continue
        if seq && isa(p.op, Par)
            set!(output_register(f,p),:incr,true)
        end
        for i in input_registers(f,p)
            if get(i,:grad) && inc!(i,:fanout) > 1
                set!(i,:incr,true)
            end
        end
    end
end

# Note about sparse arrays:
# If input x is sparse (example: matrix of one-hot word columns)
# and the first op is dot(w,x):
# w will be dense (example: word embedding matrix)
# y = w * x and all following forw outputs will be dense.
# Going back we have dw = dy * x' and dx = w' * dy
# We would like to keep dw sparse.
# The only sparse matrices are the dw and iw for w that dot sparse inputs.
# Regular sparse x is a right csc input.
# NCE adds noise and ygold matrices as left csr inputs.
# TODO: cleanup this mess: more generic code? more versatile sparse type?

function inittype2(f::Net)
    for r in registers(f)
        r.diftype = (gpu() ? CudaArray : Array)
        r.tmptype = (gpu() ? CudaArray : Array)
    end
    for p in instructions(f)
        isa(p.op, Dot) || continue
        (a, b) = input_registers(f, p)
        if issparse(a.out0) && issparse(b.out0)
            error("Dot of two sparse matrices")
        elseif issparse(b.out0) && get(a,:grad) # y = w * x  with sparse x in rnnlm
            if get(a,:incr)
                a.diftype = gpu() ? CudaArray : Array
                a.tmptype = gpu() ? CudaSparseMatrixCSRU : SparseMatrixCSC
            else
                a.diftype = gpu() ? CudaSparseMatrixCSR : SparseMatrixCSC
            end
        elseif issparse(a.out0) && get(b,:grad) # rw = r * w  with sparse r in nce
            if get(b,:incr)
                b.diftype = gpu() ? CudaArray : Array
                b.tmptype = gpu() ? CudaSparseMatrixCSCU : SparseMatrixCSC
            else
                b.diftype = gpu() ? CudaSparseMatrixCSC : SparseMatrixCSC
            end
        end
    end
end

### DEAD CODE

# function inittype2(f::Net)
#     fill!(f.sparse, nothing)
#     for o=1:length(f.op)
#         if isa(f.op[o], Dot)
#             (i,j) = f.inputs[o]
#             if issparse(f.out0[i]) && issparse(f.out0[j])
#                 error("Dot of two sparse matrices")
#             elseif issparse(f.out0[j])
#                 f.sparse[i] = f.toincr[i] ? CudaSparseMatrixCSRU : CudaSparseMatrixCSR # TODO: clean up this mess!
#             elseif issparse(f.out0[i])
#                 f.sparse[j] = f.toincr[i] ? CudaSparseMatrixCSCU : CudaSparseMatrixCSC
#             end
#         end
#     end
# end

# function finddif(f::Net, n)
#     dif0 = nothing
#     if !f.toincr[n]
#         if length(f.outputs[n]) == 0
#             # There are no outputs to try overwriting, this may happen with childless ops
#         elseif length(f.outputs[n]) == 1
#             o = f.outputs[n][1]     # first try overwriting the output dif
#             if (o > n
#                 && !f.toincr[o]
#                 && overwrites(f.op[o])
#                 && (f.dif0[o]!=nothing)
#                 && size(f.dif0[o]) == size(f.out0[n]))
#                 dif0 = f.dif0[o]
#             end
#         else
#             error("toincr==false for multi output op")
#         end
#     end
#     if dif0 == nothing
#         et = eltype(f.out0[n])
#         sz = size(f.out0[n])
#         dif0 = (
#                 ### This is sparse non-incremental dw: can't really have dense without 
#                 # rewriting CUSPARSE.csrmm to take sparse matrix in second position.  As
#                 # it stands, we'd have to transpose all three matrices: dw = dy * x' -> dw' = x * dy'
#                 (f.sparse[n]!=nothing) && !f.toincr[n] && !gpu() ? spzeros(et, sz...) :
#                 (f.sparse[n]!=nothing) && !f.toincr[n] && gpu()  ? (f.sparse[n])(spzeros(et, sz...)) : # t:12.38
#                 ### CSRU speed 20% slower than CSR on mnist (atomicAdd conflicts?), also cannot compute vecnorm.
#                 # f.sparse[n] && !f.toincr[n] && gpu()  ? CudaSparseMatrixCSRU(et, sz...) : # t:14.69
#                 ### Uncomment this if you want sparse incremental dw:
#                 ### Speed similar to dense on rnnlm, less memory, cannot compute vecnorm.
#                 # f.sparse[n] && f.toincr[n] && gpu()   ? ArrayAccumulator(et, sz) :
#                 # f.sparse[n] && f.toincr[n] && !gpu()  ? ArrayAccumulator(et, sz) :
#                 gpu() ? fill!(CudaArray(et, sz),0) : zeros(et, sz))
#     end
#     return dif0
# end

# function findtmp(f::Net, n)
#     tmp = nothing
#     for i=n+1:length(f.op)
#         if ((f.tmp[i]!=nothing) &&
#             (size(f.tmp[i]) == size(f.dif0[n])) &&
#             (f.sparse[n]==nothing || typeof(f.tmp[i]) <: f.sparse[n]))
#             tmp = f.tmp[i]
#             break
#         end
#     end
#     if tmp == nothing
#         et = eltype(f.out0[n])
#         sz = size(f.out0[n])
#         tmp = (
#                # CSRU is 5% faster if no atomic op conflicts (rnnlm), 
#                # but significantly slower when there are lots of conflicts (mnist)
#                # Not worth the risk until I implement uniq for CSRU
#                # Seems significantly faster on s2s, putting csru back on
#                gpu() && f.sparse[n]!=nothing ? (f.sparse[n])(et, sz...) :
#                # gpu() && f.sparse[n] ? CudaSparseMatrixCSR(spzeros(et, sz...)) : 
#                !gpu() && f.sparse[n]!=nothing ? spzeros(et, sz...) :
#                gpu() ? CudaArray(et, sz) : 
#                Array(et, sz))
#     end
#     return tmp
# end


# if isa(f.op[i], Input) && issparse(f.out0[i])
#     for o in f.outputs[i]
#         if isa(f.op[o], Dot)
#             w = f.inputs[o][1]
#             @assert isa(f.op[w], Par)
#             f.sparse[w] = true
#         end
#     end
# end


# # instead of changing the ops, record the eltype in net and use it when generating stuff.
#     # for o in f.op
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

# function netinit(f::Net)
#     initout(f, inputs...)
#     initdif(f, inputs...)
#     inittmp(f, inputs...)
# end

# """

# initout(f::Net, inputs...) initializes f.out[n] which points to the
# array holding the output of op[n].  f.out0[n] holds the actual array,
# allowing f.out[n] point to stack elements when necessary during back
# calculation.  We optimize memory use by sharing arrays when we can.

# """
# function initout(f::Net)
#     N = length(f.op)
#     index = zeros(Int,N)          			# index==0 represents need for new register
#     for n=1:N                    
#         overwrites(f.op[n]) || continue                 # see if we can reuse our input for output
#         f.tosave[n] && continue                         # a saved register should only be written by op[n]
#         i = f.inputs[n][1]                              # see if we can overwrite the first input
#         f.tosave[i] && continue                         # do not overwrite if you are going to save for back
#         ow = true; k = n                                # TODO: checkout other inputs if this one fails
#         while true                                      # see if anybody else uses i before its next update
#             k = mod1(k+1, N)                            # TODO: This is suboptimal, gives 13 regs for LSTM
#             in(i, f.inputs[k]) && (ow = false; break)   # Should look for existing regs no longer used
#             k == i && break                             # this is nontrivial because the sizes are unknown
#             k == N && i > N && break                    # but can possibly be inferred
#         end                                             # fencepost check: N==1:OK, i==n:OK, i>N:OK 
#         ow && (index[n]=i)
#     end
#     f.out0 = cell(N)
#     for n=1:N                                           # index==0 ops get their own registers
#         index[n]==0 || continue 
#         f.out0[n] = Any[]
#     end
#     for n=1:N                                             # other ops will overwrite existing registers
#         index[n]==0 && continue
#         k = index[n]
#         while index[k]!=0; k=index[k]; end
#         f.out0[n] = f.out0[k]
#     end
#     f.out = fill!(cell(N), nothing)                       # use nothing to represent identity element: 0 for add, 1 for mul
# end

# """
# initdif(f::Net) initializes f.dif[n], which is the loss gradient of the last output of op[n].
# f.dif0[n] holds the actual array, f.dif[n] can point to this or be 'nothing' representing the zero matrix.
# """
# function initdif(f::Net)
#     # TODO: this is not optimal for add: no need for two copies of dy when neither input is overwriting.
#     # TODO: tmp register to replace dif1 and serve other needs

#     index = zeros(Int,nops(f))                                  # index==0 represents need for new register
#     for n=1:nops(f)                                             # find out if back(op[n]) can overwrite dif[n]
#         overwrites(f.op[n]) || continue                         # overwrites means potentially both forw x<-y and back dy<-dx
#         f.toincr[n] && continue                                  # don't share dif[n] with multi-output
#         n == nops(f) && continue                                # don't share last output, dy may come in sparse
#         for i in f.inputs[n]
#             f.toincr[i] && continue                              # don't share dif[n] with multi-output
#             index[i]==0 || error("should not happen")
#             index[i]=n; break                                   # op[n] will overwrite dif[n] to get dif[i]
#         end
#     end
#     f.dif0 = cell(nops(f))
#     for n=1:nops(f)
#         index[n]==0 || continue
#         f.dif0[n] = Any[]
#     end
#     for n=1:nops(f)
#         index[n]==0 && continue
#         k = index[n]
#         while index[k]!=0; k=index[k]; end
#         f.dif0[n] = f.dif0[k]
#     end
#     f.dif = fill!(cell(nops(f)), nothing)
# end

# """
# inittmp(f::Net) initializes f.tmp[n] which is an extra array for:
# (1) incremental updates if f.toincr[n].
# (2) for forw/back ops that need extra space (TODO)
# """
# function inittmp(f::Net)
#     # TODO: this is inefficient, could use a single tmp for each size.
#     f.tmp = cell(nops(f))
#     for n=1:length(f.tmp)
#         f.tmp[n] = f.toincr[n] ? Any[] : nothing
#     end
# end

# ### DEAD CODE:


#     # initop(f, a...)
#     # initinputs(f, a...)
#     # @assert length(f.op)==length(f.inputs)
#     # initninputs(f)
#     # initparams(f)
# # function initstack(f::Net)
# #     f.stack = Any[]
# #     f.sp = 0
# # end

# #    initstack(f)

# # f.op[n] is the n'th operation in the net
# # The user specifies the operations in the MLP constructor arguments
# # If an argument is another MLP, its operations are spliced in

# # function initop(f::Net, a...)
# #     f.op = Op[]
# #     for ai in a
# #         isa(ai,Tuple) && (ai=ai[1])
# #         isa(ai,Op) ?  push!(f.op, ai) :
# #         isa(ai,Net)   ?  append!(f.op, ai.op) :
# #         error("Bad op: $ai")
# #     end
# # end

# # f.inputs[n] is an array of k indices for the inputs of the n'th op.
# # k is typically 1 but can be 2 for Add2 and Mul2 operations
# # index i<=nops(f) indicates output of f.op[i], i>nops(f) indicates network inputs
# # By default an op takes the results of the previous k op outputs
# # (or network inputs for initial ops)
# # The user can override this by providing a tuple argument for an op
# # The first element of the tuple is the op, the rest are user indices for inputs
# # userindex j>0 indicates output of userarg[j], j<=0 indicates network input 1-j.

# # function initinputs(f::Net, a...)
# #     newindex = Array(Int, length(a))
# #     lastindex = 0
# #     for i=1:length(a)
# #         ai = isa(a[i],Tuple) ? a[i][1] : a[i]
# #         lastindex += (isa(ai,Op) ? 1 : length(ai.op))
# #         newindex[i] = lastindex
# #     end
# #     f.inputs = Any[]
# #     for i=1:length(a)
# #         ai = isa(a[i],Tuple) ? a[i][1] : a[i]
# #         bi = isa(a[i],Tuple) ? a[i][2:end] : ((i-ninputs(ai)):(i-1))
# #         length(bi) == ninputs(ai) || error("Wrong number of inputs for $i:$(typeof(ai))")
# #         if isa(ai, Op)
# #             push!(f.inputs, map(j->(j>0 ? newindex[j] : nops(f)+1-j), Int[bi...]))
# #         else
# #             j0 = length(f.inputs)
# #             for aii in ai.inputs
# #                 push!(f.inputs, map(j->(j<=nops(ai) ? j+j0 : (j=bi[j-nops(ai)]; j>0 ? newindex[j] : nops(f)+1-j)), aii))
# #             end
# #         end
# #     end
# # end

# # f.ninputs is the number of inputs the whole MLP expects
# # indices i>nops(f) in f.inputs refer to network inputs

# # function initninputs(f::Net)
# #     n = 0
# #     for ai in f.inputs
# #         for aij in ai
# #             aij - nops(f) > n && (n = aij - nops(f))
# #         end
# #     end
# #     f.ninputs = n
# # end

# # f.params points to all op parameters

# # function initparams(f::Net)
# #     f.params = Any[]
# #     for o in f.op
# #         if isa(o,par) && isupdated(o)
# #             push!(f.params, o)
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

# # initforw(f::Net,x...) is called at the beginning of a sequence or
# # before processing a stand-alone item.  It is not called between
# # elements of a sequence.  It allocates and/or resizes f.out0.

# # function initforw2(f::Net, inputs...; keepstate=false, a...)
# #     f.dbg && display((:initforw0,keepstate,vecnorm0(f.out),vecnorm0(f.stack[1:f.sp])))
# #     f.sp == 0 || error("Stack corruption")
# #     length(inputs) == ninputs(f) || error("Wrong number of inputs")
# #     out = fill!(cell(length(f.out0)), nothing)          # TODO: (minor) can we get rid of alloc
# #     lastinput = 0
# #     while findfirst(out,nothing) > 0                            # allow multiple passes for size inference
# #         nnothing = count(x->(x==nothing), out)
# #         for n = 1:nops(f)
# #             if out[n] != nothing
# #                 continue
# #             elseif isa(f.op[n], Input)
# #                 out[n] = initarray(f.out0, n, inputs[lastinput += 1])
# #             elseif isempty(f.inputs[n])                         # par, rnd, con
# #                 s = psize(f, n)                                 # TODO: write psize inference
# #                 s == nothing && continue                        # may happen first pass
# #                 out[n] = initarray(f.out0, n, inputs[1], s; dense=true)
# #             else
# #                 s = ysize(f.op[n], out[f.inputs[n]]...)
# #                 s == nothing && continue # may happen with recurrent connections
# #                 out[n] = initarray(f.out0, n, out[f.inputs[n]][1], s; dense=true)
# #             end
# #         end
# #         nnothing == count(x->(x==nothing), out) && error("Cannot determine size of array $(findfirst(out,nothing))")
# #     end
# #     # We recover or reset f.out:
# #     keepstate ? copy!(f.out, f.out0) : fill!(f.out, nothing) # TODO: f.out[i] = arr(f.out0[i]) elsewhere..
# #     f.dbg && display((:initforw1,keepstate,vecnorm0(f.out),vecnorm0(f.stack[1:f.sp])))
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

#             # elseif in(nothing, dims[f.inputs[n]])
#             #     continue
#             # else
#             # end

#             # if isempty(f.inputs[n])                     # happens with par, rnd, con, input
#             #     elseif prod(size(f.op[n])) > 0
#             #         dims[n] = size(f.op[n])
#             #     end
#             # end
#             # if isa(op, Input)
#             #     dims[n] = size(inputs[lastinput += 1])
#             # elseif isa(op, Par) && prod(op.dims) > 0
#             #     dims[n] = op.dims
#             # elseif isa(op, Dot)
#             #     (i1,i2) = f.inputs[n]
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
#             #         setdims!(f.op[i1], (d1[1:end-1]..., m2))
#             #     elseif m2 == 0
#             #         length(d2) == 2 || error("Cannot infer size of tensor")
#             #         setdims!(f.op[i2], (n1, d2[2]))
#             #     elseif n1 == m2
#             #         dims[n] = (m1,n2)
#             #     else
#             #         error("Dot size mismatch")
#             #     end
#             # end
#             # # input, par, dot, add, mul, actf, loss, conv, pool

# function initback0(f::Net, dy; a...)
#     N = length(f.op)
#     firstwrite = zeros(Int, N)                          # dif[n] is read at t=n, find out when it is first written.
#     # TODO: dif[N] is special!
#     for n=N:-1:1
#         f.toback[n] || continue
#         k = n
#         while true
#             in(n, f.inputs[k]) && (firstwrite[n] = k; break)
#             k = mod1(k-1, N)
#         end
#     end
#     return firstwrite
#     for n=N:-1:1
#         nsparse = false # TODO
#         f.toback[n] || continue
#         f.dif[n] = finddif(f, n, nsparse)
#         f.toincr[n] && (f.tmp[n] = findtmp(f, n, nsparse))
#     end
# end

# function finddif(f::Net, n, nsparse)
#     isa(f.op[n], Par) && return                         # par difs should persist across iterations
#     N = length(f.op)
#     k = n; nw1 = 0                                      # dif[n] is read at t=n, find out when it is first written.
#     while true
#         in(n, f.inputs[k]) && (nw1 = k; break)          # it could potentially be written at t=n right after reading
#         k = mod1(k-1, N)
#     end
#     @assert nw1 > 0                                     # dif[n] will be busy from nw1 to n going back cyclical
#     free = nothing                                      # we need dif[i] that is free during this period
#     for i = n+1:N                                       # note that dif[i] may already be shared to some dif[j]
#         isa(f.op[i], Par) && return
#         size(f.dif[i]) == size(f.out0[n]) || continue
#         issparse(f.dif[i]) == nsparse || continue
        
#     end
#     return free
# end

# initback(f::Net, dy) called at the beginning of a sequence or a
# stand-alone item, never between elements of a sequence.

# function initback1(f::Net, dy; seq=false, a...)
#     fill!(f.dif, nothing)                           # why? (TODO)
#     for n=1:length(f.dif0)
#         y = (n==nops(f) && dy!=nothing ? dy : f.out0[n])
#         initarray(f.dif0, n, y; dense=true) # x and dw may be sparse, dx and w always dense
#         if f.toincr[n]
#             initarray(f.tmp, n, f.dif0[n])
#         end
#     end
#     for n=1:length(f.dif0)
#         f.toincr[n] && fill!(f.dif0[n], 0)           # zeroed by back at every item in sequence
#     end
#     if seq
#         for w in params(f)       # zeroed only once at the beginning of the sequence
#             isdefined(w,:diff) && isdefined(w,:inc) && fill!(w.diff,0)
#         end
#     end
# end

# function initback0(f::Net, dy, dx...; a...)
#     N = length(f.op)
#     dxback = falses(N)
#     if !isempty(dx)                                     # TODO: what if this changes after net init?
#         lastinput = 0
#         for n=1:N
#             isa(f.op[n], Input) || continue
#             dx[lastinput += 1] == nothing || (dxback[n] = true)
#         end
#     end
#     for n=N:-1:1
#         nsparse = difsparse(f, dy, n)
#         f.toback[n] || dxback[n] || continue
#         # f.dif0[n] = finddif(f, n, nsparse)  # TODO-OPTIMIZATION
#         f.dif0[n] = newarray(gpu(), nsparse, eltype(f.out0[n]), size(f.out0[n]))
#         if f.toincr[n]    # TODO: what if this changes after init?
#             # f.tmp[n] = findtmp(f, n, nsparse) # TODO-OPTIMIZATION
#             f.tmp[n] = newarray(gpu(), nsparse, eltype(f.out0[n]), size(f.out0[n]))
#         end
#     end
# end

        # else                    # otherwise find one with matching size
        #     for k=o+1:length(f.op)
        #         if (!f.toincr[k]
        #             && (f.dif0[ k]!=nothing)
        #             && size(f.dif0[k]) == size(f.out0[n])
        #             && stype(f.dif0[k]) == st
        #             && f.outputs[k][1] > k)
        #             dif0 = f.dif0[k]
        #             # TODO: However we need to check and see if this has been used between n..o
        #             # Also we don't know whether o > n for sure
        #             # This all needs more thinking
        #             break
        #         end
        #     end
# should be part of reset:
    # fill!(f.dif, nothing)
    # for n=1:length(f.op)
    #     (f.dif0[ n]!=nothing) && f.toincr[n] && fill!(f.dif0[n], 0)
    # end

# for n=length(f.op):-1:1
#     f.toback[n] || continue # we mix initback and initback0 here because if getdx or seq changes we may need to alloc new arrays.
#     if (f.dif0[n]!=nothing) # TODO: implement batch size change
#         @assert (issimilar2(f.dif0[n], f.out0[n]) ) #TODO: && issparse(f.dif0[n])==(f.sparse[n]!=nothing)
#     else
#         f.dif0[n] = finddif(f, n)
#         # f.dif[n] = nothing # Leave this to reset! otherwise s2s does not work
#     end
#     if f.toincr[n]
#         if (f.tmp[n]!=nothing)
#             @assert (issimilar2(f.tmp[n], f.out0[n]) && issparse(f.tmp[n])==(f.sparse[n]!=nothing))
#         else
#             f.tmp[n] = findtmp(f, n)
#             # fill!(f.dif0[n], 0)  # This will break s2s best if finddif zeros during alloc
#         end
#     end
# end
