# For each loss function we have 24 definitions!:
# CPU[4]: { 2arg, 3arg } x { sparse, dense }
# GPU[20]: { 2arg, 3arg } x { sparse, dense } x { Julia, JuliaMixed, CUDA, C32, C64 }

"""
```
loss = softloss(ypred,ygold)
softloss(ypred,ygold,ygrad)
```

Cross entropy loss to use after a softmax output (see the `soft`
activation function).  ypred has the normalized probabilities output
by the model.  ygold has normalized probabilities from the answer key.
For general arrays normalization is across the last dimension,
i.e. sum(y[:,...,:,i])==1.

The two argument version calculates the loss value:

loss = -Σ ygold[i,j] log ypred[i,j]

The three argument version calculates the gradient of the loss wrt ypred:

ygrad[i,j] = 1 - ygold[i,j]/ypred[i,j]

If a certain column needs to be ignored (e.g. it is padding for a
minibatch), set ygold[:,j]=0.  This will eliminate its contribution to
loss.  The gradient calculation will detect this and set ygrad[:,j]=0.

For the math let us use p for ygold, q for ypred, and ∂J/∂qk for the
derivative of the loss wrt ypred[k]:
```
J = -Σ pi log qi                        ;; loss function
  = -Σ pi log (qi/Σqj)                  ;; normalization explicit
  = (-Σ pi log qi) + Σ pi log Σ qj
  = (-Σ pi log qi) + log Σ qj

∂J/∂qk = -pk/qk + (1/Σ qj)
       = -pk/qk + 1
```
"""
function softloss(ypred::Array, ygold::Array, ygrad::Array; mask=nothing, o...)
    @assert size(ypred)==size(ygold)==size(ygrad)
    (yrows,ycols) = size2(ygrad)
    batchsize = (mask == nothing ? ycols : sum(mask))
    for i=1:length(ygrad)
        ygrad[i] = (mask == nothing || mask[1 + div((i-1),yrows)] != 0 ?
                    ((ypred[i]-ygold[i])/ypred[i])/batchsize : 0)
    end
end

@gpu function softloss{T}(ypred::CudaArray{T}, ygold::CudaArray{T}, ygrad::CudaArray{T}; mask=C_NULL, o...)
    (yrows,ycols) = size2(ygrad)
    if mask != C_NULL
        batchsize = sum(mask)
        mask = convert(CudaArray, mask)
    else
        batchsize = ycols
    end
    T <: Float32 ? ccall((:softlossback32,libknet),Void,(Cint,Cint,Cint,Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cuchar}, Ptr{Cfloat}), 
                         yrows,ycols,batchsize,ypred,ygold,mask,ygrad) :
    T <: Float64 ? ccall((:softlossback64,libknet),Void,(Cint,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cuchar},Ptr{Cdouble}),
                         yrows,ycols,batchsize,ypred,ygold,mask,ygrad) : error()
    gpusync()
    #@dbg println((:softlossbackgpudense,vecnorm0(ypred,ygold,ygrad),mask==C_NULL?mask:convert(Vector{Int},mask)))
    return ygrad
end

@gpu softloss(ypred::CudaArray, ygold::Array, ygrad::CudaArray; o...)=softloss(ypred, CudaArray(ygold), ygrad; o...)

function softloss(ypred::Array, ygold::SparseMatrixCSC, ygrad::Array; mask=nothing, o...)
    batchsize = (mask == nothing ? size(ygold,2) : sum(mask))
    col = 1             # Column i is in colptr[i]:(colptr[i+1]-1)
    for nz = 1:nnz(ygold)
        while nz > ygold.colptr[col+1]-1; col += 1; end
        if mask == nothing || mask[col] != 0
            ygoldi = ygold.nzval[nz]
            row = ygold.rowval[nz]
            i = (col-1) * size(ypred,1) + row
            ygrad[i] = (1-ygoldi/ypred[i])/batchsize
        else
            ygrad[i] = 0
        end
    end
    return ygrad
end

@gpu function softloss{T}(ypred::CudaArray{T}, ygold::CudaSparseMatrixCSC{T}, ygrad::CudaArray{T}; mask=C_NULL, o...)
    (yrows,ycols) = size2(ygrad)
    if mask != C_NULL
        batchsize = sum(mask)
        mask = convert(CudaArray, mask)
    else
        batchsize = ycols
    end
    T <: Float32 ? ccall((:softlossback32csc,libknet),Void,(Cint,Cint,Cint,Ptr{Cfloat}, Cint,Ptr{Cfloat}, Ptr{Cint},Ptr{Cint},Ptr{Cuchar},Ptr{Cfloat}), 
                         yrows,ycols,batchsize,ypred,ygold.nnz,ygold.nzVal,ygold.rowVal,ygold.colPtr,mask,ygrad) :
    T <: Float64 ? ccall((:softlossback64csc,libknet),Void,(Cint,Cint,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cuchar},Ptr{Cdouble}),
                         yrows,ycols,batchsize,ypred,ygold.nnz,ygold.nzVal,ygold.rowVal,ygold.colPtr,mask,ygrad) : error()
    gpusync()
    #@dbg println((:softlossbackgpusparse,vecnorm0(ypred,ygold,ygrad),mask==C_NULL?mask:convert(Vector{Int},mask)))
    return ygrad
end
@gpu softloss(ypred::CudaArray, ygold::SparseMatrixCSC, ygrad::CudaArray; o...)=softloss(ypred, CudaSparseMatrixCSC(ygold), ygrad; o...)


### The two argument version is for loss calculation:

function softloss(ypred::Array, ygold::Array; mask=nothing)
    @assert size(ypred)==size(ygold)
    (yrows,ycols) = size2(ygold)
    logp=zero(Float64)
    for i=1:length(ygold)
        (mask==nothing || mask[1 + div((i-1),yrows)] != 0) &&
        ygold[i] > 0 &&
        (logp += (ygold[i]*log(ypred[i])))
    end
    batchsize = (mask == nothing ? ycols : sum(mask))
    return -logp/batchsize
end

@gpu function softloss{T}(ypred::CudaArray{T}, ygold::CudaArray{T}; tmp=nothing, mask=C_NULL, o...)
    (yrows,ycols) = size2(ygold)
    if mask != C_NULL
        batchsize = sum(mask)
        mask = convert(CudaArray,mask)
    else
        batchsize = ycols
    end
    ly = (tmp == nothing ? similar(ypred) : tmp) # TODO: get rid of alloc
    T <: Float32 ? ccall((:softloss32,libknet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cuchar},Ptr{Cfloat}),
                         yrows,ycols,ypred,ygold,mask,ly) :
    T <: Float64 ? ccall((:softloss64,libknet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cuchar},Ptr{Cdouble}),
                         yrows,ycols,ypred,ygold,mask,ly) : error()
    loss = CUBLAS.asum(ly)/batchsize
    #@dbg println((:softlosslossgpudense,loss,vecnorm0(ly,ypred,ygold),mask==C_NULL?mask:convert(Vector{Int},mask)))
    #@dbg push!(DBGSTACK, map(copy,(ly,ypred,ygold,mask)))
    ly === tmp || free(ly)
    gpusync()
    return loss
end

@gpu softloss(ypred::CudaArray, ygold::Array; o...)=softloss(ypred, CudaArray(ygold); o...)

function softloss(ypred::Array, ygold::SparseMatrixCSC; mask=nothing, o...)
    (yrows,ycols) = size(ygold)
    logp=zero(Float64)
    col = 1             # Column i is in colptr[i]:(colptr[i+1]-1)
    for nz = 1:nnz(ygold)
        while nz > ygold.colptr[col+1]-1; col += 1; end
        mask!=nothing && mask[col]==0 && continue
        ygoldi = ygold.nzval[nz]
        row = ygold.rowval[nz]
        i = (col-1) * yrows + row
        logp += (ygoldi * log(ypred[i]))
    end
    batchsize = (mask == nothing ? ycols : sum(mask))
    return -logp/batchsize
end

@gpu function softloss{T}(ypred::CudaArray{T}, ygold::CudaSparseMatrixCSC{T}; mask=C_NULL, tmp=nothing, o...)
    (yrows,ycols) = size(ygold)
    if mask!=C_NULL
        batchsize = sum(mask)
        mask=convert(CudaArray,mask)
    else
        batchsize = ycols
    end
    ly = (tmp == nothing ? similar(ygold.nzVal) : tmp) # TODO: get rid of alloc
    length(ly) >= nnz(ygold) || error("not enough temp space")
    T <: Float32 ? ccall((:softloss32csc,libknet),Void,(Cint,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cuchar},Ptr{Cfloat}),
                         yrows,ycols,ypred,ygold.nnz,ygold.nzVal,ygold.rowVal,ygold.colPtr,mask,ly) :
    T <: Float64 ? ccall((:softloss64csc,libknet),Void,(Cint,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cuchar},Ptr{Cdouble}),
                         yrows,ycols,ypred,ygold.nnz,ygold.nzVal,ygold.rowVal,ygold.colPtr,mask,ly) : error()
    loss = CUBLAS.asum(nnz(ygold),ly,1)/batchsize
    ly === tmp || free(ly)
    gpusync()
    #@dbg println((:softlosslossgpusparse,loss,vecnorm0(ypred,ygold.nzVal),mask==C_NULL?mask:convert(Vector{Int},mask)))
    return loss
end

@gpu softloss(ypred::CudaArray, ygold::SparseMatrixCSC; o...)=softloss(ypred, CudaSparseMatrixCSC(ygold); o...)


### QUADLOSS:
# TODO: missing masks

# Quadratic loss:
# y stores the model output.
# ygold is the desired output.
# Overwrites ygrad with the gradient of quadratic loss wrt y, i.e. y-ygold
# J = (1/2)*sum((y-ygold)^2)
# dJ/dy = y-ygold

# This is cpu/gpu generic, the rest is dead code:

function quadloss(y::BaseArray, ygold::BaseArray, ygrad::BaseArray)
    @assert size(y)==size(ygold)==size(ygrad)
    ycols = ccount(y)
    ygrad === ygold || copy!(ygrad, ygold) # TODO: avoid copy if possible
    scale!(-1/ycols, ygrad)
    axpy!(1/ycols, y, ygrad)
    gpusync()
    return ygrad
end

function quadloss(y::BaseArray, ygold::BaseArray)
    ytemp = similar(y)         # TODO: avoid alloc
    copy!(ytemp, ygold)
    axpy!(-1, y, ytemp)
    qloss = vecnorm(ytemp)^2/(2*ccount(y))
    free(ytemp)
    gpusync()
    return qloss
end

### ZERO-ONE LOSS

function zeroone(ypred::Array, ygold::Array)
    (yrows,ycols) = size2(ypred)
    cost = 0
    tmin = typemin(eltype(ypred))
    for j=1:ycols
        (cz,cy,ymax,zmax) = (0,0,tmin,tmin)
        i1=(j-1)*yrows+1; i2=j*yrows
        for i=i1:i2
            ypred[i] > ymax && ((cy,ymax) = (i,ypred[i]))
            ygold[i] > zmax && ((cz,zmax) = (i,ygold[i]))
        end
        (cz != cy) && (cost += 1)
    end
    return cost/ycols
end

zeroone(ypred,ygold)=zeroone(convert(Array,ypred),convert(Array,ygold))


### DEAD CODE

# TODO: ALL THESE NEED TESTING

# Loss Layers
# TODO: get rid of l.y in documentation

# abstract Loss <: Op

# ninputs(::Loss)=1
# infersize(::Loss,dims)=(dims==nothing ? nothing : (dims,dims))
# overwrites(::Loss)=true
# back_reads_x(::Loss)=false
# back_reads_y(::Loss)=true

# # Loss has slightly different input/output behavior compared to regular layers:
# # forw only records the outgoing y.
# # back takes dy, the desired output, and returns the loss gradient wrt y
# # loss takes dy, the desired output, and returns a loss value

# for (ltype, lback, lloss, lname) in 
#     ((:QuadLoss, :quadlossback, :quadlossloss, :quadloss),
#      (:SoftLoss, :softlossback, :softlossloss, :softloss),
#      (:LogpLoss, :logplossback, :logplossloss, :logploss),
#      (:XentLoss, :xentlossback, :xentlossloss, :xentloss),
#      (:PercLoss, :perclossback, :perclossloss, :percloss),
#      (:ScalLoss, :scallossback, :scallossloss, :scalloss))
#     @eval begin
#         type $ltype <: Loss; end

#         # $lname() = $ltype()

#         function forw(l::$ltype, x, y; o...)
#             size(x) == size(y) || error(map(summary,(x,y)))
#             (y===x ? y : copy!(y,x)) # TODO: is this copy necessary?
#         end

#         function back(l::$ltype, dy, dx; y=nothing, o...)
#             size(y)==size(dy) || error(map(summary,(dy,y)))
#             dx == nothing && return
#             size(y)==size(dx) || error(map(summary,(dx,y)))
#             $lback(y,dy,dx; o...)
#         end

#         function loss(l::$ltype, dy, y; o...)
#             size(y)==size(dy) || error(map(summary,(y,dy)))
#             $lloss(y,dy; o...)
#         end

#     end
# end


# # function softlossloss(y::Array, dy::Array; o...)
# #     cost=zero(Float64)
# #     for i=1:length(dy)
# #         dy[i]>0 && (cost -= (dy[i]*log(y[i])))
# #     end
# #     return cost/ccount(dy)
# # end

# # function softlossback(y::Array, dy::Array, dx::Array; o...)
# #     nx=ccount(dx)
# #     for i=1:length(dx)
# #         dx[i] = ((y[i]-dy[i])/y[i])/nx
# #     end
# #     return dx
# # end


# # Convenience op combining soft and softloss:

# # softmax()=quote
# #     x = input()
# #     y = soft(x)
# #     z = softloss(y)
# # end

# # function quadlossloss(y::Array, dy::Array; o...)
# #     cost=zero(Float64)
# #     for i=1:length(dy) 
# #         cost += (y[i]-dy[i])^2
# #     end
# #     0.5*cost/ccount(dy)
# # end

# # @gpu function quadlossloss(y::CudaArray, dy::CudaArray; tmp=nothing, o...)
# #     tmp == nothing && (tmp = similar(y)) # t:87/472
# #     copy!(tmp, y)                        # t:29/472
# #     axpy!(-1, dy, tmp)                   # t:24/472
# #     vecnorm(tmp)^2/(2*ccount(y))         # t:330/472
# # end

# # # quadlossback(y::Array, dy::Array, dx::Array=dy; o...)=(nx=ccount(dx); for i=1:length(dx); dx[i] = (y[i]-dy[i])/nx; end; dx)
# # # @gpu quadlossback(y::CudaArray, dy::CudaArray, dx::CudaArray=dy; o...)=(dx===dy||copy!(dx,dy); cudnnTransformTensor(1/ccount(y), y, -1/ccount(y), dx); dx)  ## cudnnTransformTensor is buggy

# # quadlossback(y, dy, dx=dy; o...)=(dx===dy||copy!(dx,dy); scale!(-1/ccount(y), dx); axpy!(1/ccount(y), y, dx); dx)

# ### LOGPLOSS:

# # Cross entropy loss to use after the Logp layer.
# # l.y should be normalized log probabilities output by the model.
# # p has normalized probabilities from the answer key.
# # Normalization is across the last dimension, i.e. sum(p[:,...,:,i])==1
# # Overwrites p with the gradient of the loss wrt y, i.e. exp(y)-p:
# #
# # dy = sum(exp(y))   ;; normalization constant (should be 1 here)
# # q = exp(y)/dy      ;; model probabilities
# # logq = y - logz   ;; model (normalized) log prob
# # dlogz/dy = q      
# #
# # J = (1/N) Σ[nc] -p[nc]*logq[nc]  ;; n=1..N: instance, c=1..C: class
# #   = (1/N) Σ[nc] -p[nc]*(y[nc]-logz[n])
# #   = (1/N) ((Σ[n] logz[n]) - (Σ[nc] p[nc]*y[nc]))
# #   = (1/N) (Σ[nc] -p[nc]*y[nc])   ;; all logz are 0
# #
# # dJ/dy[md] = (1/N) (q[md] - p[md])

# logplossloss(y::Array, dy::Array)=(nx = ccount(dy); cost = zero(Float64); for i=1:length(dy); cost -= (dy[i]*y[i]); end; cost/nx)
# logplossback(y::Array, dy::Array, dx::Array=dy)=(nx = ccount(dx); for i=1:length(dx); dx[i] = (exp(y[i])-dy[i])/nx; end; dx)
# @gpu (logplossback(y::CudaArray{Float32}, dy::CudaArray{Float32}, dx::CudaArray{Float32}=dy)=
#         (ccall((:logplossback32,libknet),Void,(Cint,Cdouble,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),
#                length(dy),1/ccount(dy),y,dy,dx); dx))
# @gpu (logplossback(y::CudaArray{Float64}, dy::CudaArray{Float64}, dx::CudaArray{Float64}=dy)=
#         (ccall((:logplossback64,libknet),Void,(Cint,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),
#                length(dy),1/ccount(dy),y,dy,dx); dx))


# ### XENTLOSS:

# # Cross entropy loss to use after an unnormalized layer.
# # l.y is treated as unnormalized log probabilities output by the model.
# # p has normalized probabilities from the answer key.
# # Normalization is across the last dimension, i.e. sum(p[:,...,:,i])==1
# # Overwrites p with the gradient of the loss wrt y, i.e. q-p:
# #
# # z = sum(exp(y))    ;; normalization constant
# # q = exp(y)/z       ;; model probabilities
# # logq = y - logz    ;; model (normalized) log prob
# # dlogz/dy = q      
# #
# # J = (1/N) Σ[nc] -p[nc]*logq[nc]  ;; n=1..N: instance, c=1..C: class
# #   = (1/N) Σ[nc] -p[nc]*(y[nc]-logz[n])
# #   = (1/N) ((Σ[n] logz[n]) - (Σ[nc] p[nc]*y[nc]))
# #
# # dJ/dy[md] = (1/N) (q[md] - p[md])

# function xentlossloss(y::Array, p::Array)
#     cost = zero(Float64)
#     (nd,nx) = size2(p)
#     for j=1:nx
#         i1=(j-1)*nd+1; i2=j*nd
#         z = sumpy = zero(Float64)
#         ymax = typemin(eltype(y))
#         for i=i1:i2; y[i] > ymax && (ymax = y[i]); end
#         for i=i1:i2; yi=y[i]-ymax; z += exp(yi); sumpy += p[i]*yi; end
#         cost += (log(z) - sumpy)
#     end
#     return cost/nx
# end

# function xentlossback(y::Array, p::Array, dx::Array=p)
#     (nd,nx) = size2(p)
#     for j=1:nx
#         i1=(j-1)*nd+1; i2=j*nd
#         z = zero(Float64)
#         ymax = typemin(eltype(y)) # subtract ymax for numerical stability
#         for i=i1:i2; y[i] > ymax && (ymax = y[i]); end
#         for i=i1:i2; z += exp(y[i]-ymax); end
#         for i=i1:i2; yi = exp(y[i]-ymax)/z; dx[i] = (yi - p[i])/nx; end
#     end
#     return dx
# end

# @gpu (xentlossback(y::CudaArray{Float32}, p::CudaArray{Float32}, dx::CudaArray{Float32}=p)=
#         ((nd,nx)=size2(p);ccall((:xentlossback32,libknet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),nd,nx,y,p,dx);dx))
# @gpu (xentlossback(y::CudaArray{Float64}, p::CudaArray{Float64}, dx::CudaArray{Float64}=p)=
#         ((nd,nx)=size2(p);ccall((:xentlossback64,libknet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),nd,nx,y,p,dx);dx))


# ### PERCLOSS

# # Perceptron loss function.

# # Going forward perceptron computes y=w*x and PercLoss simply records
# # the output y.  size(w)=(nc,nd) where nc is the number of classes and
# # nd is the number of x dimensions (i.e. features).  size(x)=(nd,nx)
# # where nd is the number of features and nx is the batch size.  This
# # gives us size(y)=(nc,nx) where the highest entry in each column of y
# # indicates the predicted class.

# # Going back we get a dy matrix with size(dy)=(nc,nx) where the correct
# # answer is marked with the maximum entry in each column.
# # For a given column with input x, if cz is the correct answer and cy
# # is the predicted answer, the multiclass perceptron update rule is:

# # w[cz,:] += x;  w[cy,:] -= x

# # Note that there is no update if cz==cy.

# # The mmul updates are:
# # dw = dy*x'
# # dx = w'*dy

# # So the perceptron update will be performed if we pass a dy matrix
# # back where in each column we have all zeros if the predicted answer
# # is correct, otherwise the correct answer is marked with -1 and the
# # predicted answer is marked with a +1.  The signs might be confusing,
# # this is the gradient of the loss, i.e. going in this direction will
# # increase the loss.  We will overwrite the dy matrix.

# # This update can be seen as the gradient of a perceptron loss
# # function Sum(-y[I]+y[J]) where I are the indices for the correct
# # answers, and J are the indices for predicted answers.

# function percloss{T}(ypred::Array{T}, ygold::Array{T})
#     (nc,nx) = size2(ypred)
#     cost = zero(Float64)
#     for j=1:nx
#         (cz,cy,ymax,zmax) = (0,0,typemin(T),typemin(T))
#         i1=(j-1)*nc+1; i2=j*nc
#         for i=i1:i2
#             ypred[i] > ymax && ((cy,ymax) = (i,ypred[i]))
#             ygold[i] > zmax && ((cz,zmax) = (i,ygold[i]))
#         end
#         (cz != cy) && (cost += ypred[cy]; cost -= ypred[cz])
#     end
#     return cost/nx
# end

# function perclossback{T}(ypred::Array{T}, ygold::Array{T}, dx::Array{T}=ygold)
#     (nc,nx) = size2(ypred)
#     for j=1:nx
#         (cz,cy,ymax,zmax) = (0,0,typemin(T),typemin(T))
#         i1=(j-1)*nc+1; i2=j*nc
#         for i=i1:i2
#             ypred[i] > ymax && ((cy,ymax) = (i,ypred[i]))
#             ygold[i] > zmax && ((cz,zmax) = (i,ygold[i]))
#             dx[i] = zero(T)
#         end
#         # TODO: these should be scaled 1/nx, why isn't our gradient check complaining?
#         (cz != cy) && (dx[cz] = -1; dx[cy] = 1)
#     end
#     return dx
# end

# @gpu (perclossback(ypred::CudaArray{Float32}, ygold::CudaArray{Float32}, dx::CudaArray{Float32}=ygold)=((nd,nx)=size2(ygold);ccall((:perclossback32,libknet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),nd,nx,ypred,ygold,dx);dx))
# @gpu (perclossback(ypred::CudaArray{Float64}, ygold::CudaArray{Float64}, dx::CudaArray{Float64}=ygold)=((nd,nx)=size2(ygold);ccall((:perclossback64,libknet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),nd,nx,ypred,ygold,dx);dx))


# ### SCALLOSS
# #
# # When we do structured training, gradients rather than answers come
# # back.  We should scale them using training batch size so the
# # learning rate is independent of batch size.  TODO: find a better
# # interface.  can't pass back target probabilities because network
# # output is not locally normalized.  can we pass back anything so one
# # of the existing loss functions would work?

# scallossloss(y,dy)=error("Not implemented")
# scallossback(y,dy,dx=dy)=(dx===dy||copy!(dx,dy);scale!(1/ccount(dx), dx))



# ### DEAD CODE:

#         # TODO: can we take these out and make them apply to Loss?
#         # $lloss(y::KUdense{Array}, dy::KUdense{Array}; o...)=$lloss(convert(Array,y), convert(Array,dy); o...)
#         # $lloss(y::KUdense{Array}, dy::Array; o...)=$lloss(convert(Array,y), convert(Array,dy); o...)
#         # $lloss(y::KUdense{Array}, dy::KUdense{Array}; o...)=$lloss(convert(Array,y), convert(Array,dy); o...)
#         # $lloss(y::KUdense{Array}, dy::SparseMatrixCSC; o...)=$lloss(convert(Array,y), dy; o...)
#         # @gpu $lloss{T}(y::KUdense{CudaArray,T},dy::Array{T}; o...)=$lloss(convert(CudaArray,y), convert(CudaArray,dy); o...)
#         # @gpu $lloss{T}(y::KUdense{CudaArray,T},dy::KUdense{Array,T}; o...)=$lloss(convert(CudaArray,y), convert(CudaArray,dy); o...)
#         # @gpu $lloss{T}(y::KUdense{CudaArray,T},dy::SparseMatrixCSC{T}; o...)=$lloss(convert(CudaArray,y), convert(CudaSparseMatrixCSC,dy); o...)

#         # $lback(y::KUdense{Array}, dy::KUdense{Array}, dx::KUdense{Array};o...)=($lback(convert(Array,y), convert(Array,dy), convert(Array, dx);o...); dx)
#         # $lback(y::KUdense{Array}, dy::SparseMatrixCSC, dx::KUdense{Array};o...)=($lback(convert(Array,y), dy, convert(Array, dx);o...); dx)
#         # @gpu $lback(y::KUdense{CudaArray}, dy::KUdense{CudaArray}, dx::KUdense{CudaArray};o...)=($lback(convert(CudaArray,y), convert(CudaArray,dy), convert(CudaArray, dx);o...); dx)
#         # @gpu $lback(y::KUdense{CudaArray}, dy::CudaSparseMatrixCSC, dx::KUdense{CudaArray};o...)=($lback(convert(CudaArray,y), dy, convert(CudaArray, dx);o...); dx)

#         # $lback(y::KUdense, dy::KUdense, dx::KUdense=dy)=($lback(y.arr, dy.arr, dx.arr); dx)
#         # $lloss(y,dy)=$lloss(convert(Array,y), convert(Array,dy))  # TODO: handle sparse arrays, implement gpu

# # params(::Loss)=Any[]
# # overwrites(::Loss)=true

