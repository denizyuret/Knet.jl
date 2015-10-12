# TODO: ALL THESE NEED TESTING

# Loss Layers
# TODO: get rid of l.y in documentation

abstract Loss <: Op

ninputs(::Loss)=1
infersize(::Loss,dims)=(dims==nothing ? nothing : (dims,dims))
overwrites(::Loss)=true
back_reads_x(::Loss)=false
back_reads_y(::Loss)=true

# Loss has slightly different input/output behavior compared to regular layers:
# forw only records the outgoing y.
# back takes dy, the desired output, and returns the loss gradient wrt y
# loss takes dy, the desired output, and returns a loss value

for (ltype, lback, lloss, lname) in 
    ((:QuadLoss, :quadlossback, :quadlossloss, :quadloss),
     (:SoftLoss, :softlossback, :softlossloss, :softloss),
     (:LogpLoss, :logplossback, :logplossloss, :logploss),
     (:XentLoss, :xentlossback, :xentlossloss, :xentloss),
     (:PercLoss, :perclossback, :perclossloss, :percloss),
     (:ScalLoss, :scallossback, :scallossloss, :scalloss))
    @eval begin
        type $ltype <: Loss; end

        # $lname() = $ltype()

        function forw(l::$ltype, x, y; o...)
            size(x) == size(y) || error(map(summary,(x,y)))
            (y===x ? y : copy!(y,x)) # TODO: is this copy necessary?
        end

        function back(l::$ltype, dy, dx; y=nothing, o...)
            size(y)==size(dy) || error(map(summary,(dy,y)))
            dx == nothing && return
            size(y)==size(dx) || error(map(summary,(dx,y)))
            $lback(y,dy,dx; o...)
        end

        function loss(l::$ltype, dy, y; o...)
            size(y)==size(dy) || error(map(summary,(y,dy)))
            $lloss(y,dy; o...)
        end

    end
end

### SOFTLOSS: 

# Cross entropy loss to use after the Soft layer.
# y should have normalized probabilities output by the model.
# p has normalized probabilities from the answer key.
# Normalization is across the last dimension, i.e. sum(p[:,...,:,i])==1
# Calculates the gradient of the loss wrt y, i.e. 1-p/y

# Math:
#
# J = -Σ pi log yi		;; loss function
#   = -Σ pi log (yi/Σyj)	;; should make normalization explicit
#   = (-Σ pi log yi) + Σ pi log Σ yj
#   = (-Σ pi log yi) + log Σ yj
#
# ∂J/∂yk = -pk/yk + (1/Σ yj)
#        = -pk/yk + 1
#
# dy = wx			;; dy is the input to the soft layer
# yi = (exp zi) / (Σ exp zj)	;; y is the output of the soft layer
# ∂yi/∂zk = [(i=k)(exp zi)(Σ exp zj) - (exp zi)(exp zk)] / (Σ exp zj)^2
#         = (i=k) yi - yi yk
# ∂J/∂zk = Σ (∂J/∂yi)(∂yi/∂zk)	;; derivative wrt the input dy
#        = Σ (1-pi/yi)((i=k) yi - yi yk)
#        = Σ ((i=k) yi - yi yk - (i=k) pi + pi yk)
#        = yk - pk - yk Σ (yi - pi)
#        = yk - pk


function softloss(ypred::Array, ygold::Array, ygrad::Array; o...)
    @assert size(ypred)==size(ygold)==size(ygrad)
    ycols=ccount(ygrad)
    for i=1:length(ygrad)
        ygrad[i] = ((ypred[i]-ygold[i])/ypred[i])/ycols
    end
end

@gpu softloss(ypred::CudaArray{Float32}, ygold::CudaArray{Float32}, ygrad::CudaArray{Float32}; o...)=(ccall((:softlossback32,libknet),Void,(Cint,Cdouble,Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}), length(ygold),1/ccount(ygold),ypred,ygold,ygrad); gpusync(); ygrad)
@gpu softloss(ypred::CudaArray{Float64}, ygold::CudaArray{Float64}, ygrad::CudaArray{Float64}; o...)=(ccall((:softlossback64,libknet),Void,(Cint,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),length(ygold),1/ccount(ygold),ypred,ygold,ygrad); gpusync(); ygrad)
@gpu softloss(ypred::CudaArray, ygold::Array, ygrad::CudaArray)=softloss(ypred, CudaArray(ygold), ygrad)

function softloss(ypred::Array, ygold::SparseMatrixCSC, ygrad::Array; o...)
    fill!(ygrad, 1/size(ygold,2))
    for nz = 1:nnz(ygold)
        dyi = ygold.nzval[nz]
        row = ygold.rowval[nz]
        col = 1             # Column i is in colptr[i]:(colptr[i+1]-1)
        while nz > ygold.colptr[col+1]-1; col += 1; end
        i = (col-1) * size(ypred,1) + row
        ygrad[i] *= (1-dyi/ypred[i])
    end
    return ygrad
end

@gpu softloss(ypred::CudaArray{Float32}, ygold::CudaSparseMatrixCSC{Float32}, ygrad::CudaArray{Float32}; o...)=(ccall((:softlossback32csc,libknet),Void,(Cint,Cint,Ptr{Cfloat}, Cint,Ptr{Cfloat}, Ptr{Cint},Ptr{Cint},Ptr{Cfloat}), size(ygold,1),size(ygold,2),ypred,ygold.nnz,ygold.nzVal,ygold.rowVal,ygold.colPtr,ygrad);gpusync();ygrad)
@gpu softloss(ypred::CudaArray{Float64}, ygold::CudaSparseMatrixCSC{Float64}, ygrad::CudaArray{Float64}; o...)=(ccall((:softlossback64csc,libknet),Void,(Cint,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),size(ygold,1),size(ygold,2),ypred,ygold.nnz,ygold.nzVal,ygold.rowVal,ygold.colPtr,ygrad);gpusync();ygrad)
@gpu softloss(ypred::CudaArray, ygold::SparseMatrixCSC, ygrad::CudaArray)=softloss(ypred, CudaSparseMatrixCSC(ygold), ygrad)

function softloss(ypred::Array, ygold::Array)
    @assert size(ypred)==size(ygold)
    cost=zero(Float64)
    for i=1:length(ygold)
        ygold[i] > 0 && (cost += (ygold[i]*log(ypred[i])))
    end
    return -cost/ccount(ygrad)
end

@gpu function softloss(ypred::CudaArray{Float32}, ygold::CudaArray{Float32}; tmp=nothing, o...)
    ly = (tmp == nothing ? similar(ypred) : tmp) # TODO: get rid of alloc
    ccall((:softloss32,libknet),Void,(Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),length(ygold),ypred,ygold,ly)
    loss = CUBLAS.asum(ly)/ccount(ygold)
    tmp == nothing && free(ly)
    gpusync()
    return loss
end

@gpu function softloss(ypred::CudaArray{Float64}, ygold::CudaArray{Float64}; tmp=nothing, o...)
    ly = (tmp == nothing ? similar(ypred) : tmp) # TODO: get rid of alloc
    ccall((:softloss64,libknet),Void,(Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),length(ygold),ypred,ygold,ly)
    loss = CUBLAS.asum(ly)/ccount(ygold)
    tmp == nothing && free(ly)
    gpusync()
    return loss
end

@gpu softloss(ypred::CudaArray, ygold::Array; o...)=softloss(ypred, CudaArray(ygold); o...)

function softloss(ypred::Array, ygold::SparseMatrixCSC; o...)
    cost=zero(Float64)
    for nz = 1:nnz(ygold)
        dyi = ygold.nzval[nz]
        row = ygold.rowval[nz]
        col = 1             # Column i is in colptr[i]:(colptr[i+1]-1)
        while nz > ygold.colptr[col+1]-1; col += 1; end
        i = (col-1) * size(ypred,1) + row
        cost -= (dyi * log(ypred[i]))
    end
    return cost/ccount(ygold)
end

@gpu function softloss(ypred::CudaArray{Float32}, ygold::CudaSparseMatrixCSC{Float32}; tmp=nothing, o...)
    ly = (tmp == nothing ? similar(ygold.nzVal) : tmp) # TODO: get rid of alloc
    length(ly) >= nnz(ygold) || error("not enough temp space")
    ccall((:softloss32csc,libknet),Void,(Cint,Cint,Ptr{Cfloat},Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
          size(ygold,1),size(ygold,2),ypred,ygold.nnz,ygold.nzVal,ygold.rowVal,ygold.colPtr,ly)
    loss = CUBLAS.asum(nnz(ygold),ly,1)/ccount(ygold)
    tmp == nothing && free(ly)
    gpusync()
    return loss
end

@gpu function softloss(ypred::CudaArray{Float64}, ygold::CudaSparseMatrixCSC{Float64}; tmp=nothing, o...)
    ly = (tmp == nothing ? similar(ygold.nzVal) : tmp) # TODO: get rid of alloc
    length(ly) >= nnz(ygold) || error("not enough temp space")
    ccall((:softloss64csc,libknet),Void,(Cint,Cint,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
          size(ygold,1),size(ygold,2),ypred,ygold.nnz,ygold.nzVal,ygold.rowVal,ygold.colPtr,ly)
    loss = CUBLAS.asum(nnz(ygold),ly,1)/ccount(ygold)
    tmp == nothing && free(ly)
    gpusync()
    return loss
end

@gpu softloss(ypred::CudaArray, ygold::SparseMatrixCSC; o...)=softloss(ypred, CudaSparseMatrixCSC(ygold); o...)


# function softlossloss(y::Array, dy::Array; o...)
#     cost=zero(Float64)
#     for i=1:length(dy)
#         dy[i]>0 && (cost -= (dy[i]*log(y[i])))
#     end
#     return cost/ccount(dy)
# end

# function softlossback(y::Array, dy::Array, dx::Array; o...)
#     nx=ccount(dx)
#     for i=1:length(dx)
#         dx[i] = ((y[i]-dy[i])/y[i])/nx
#     end
#     return dx
# end


# Convenience op combining soft and softloss:

# softmax()=quote
#     x = input()
#     y = soft(x)
#     z = softloss(y)
# end

### QUADLOSS:

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

# function quadlossloss(y::Array, dy::Array; o...)
#     cost=zero(Float64)
#     for i=1:length(dy) 
#         cost += (y[i]-dy[i])^2
#     end
#     0.5*cost/ccount(dy)
# end

# @gpu function quadlossloss(y::CudaArray, dy::CudaArray; tmp=nothing, o...)
#     tmp == nothing && (tmp = similar(y)) # t:87/472
#     copy!(tmp, y)                        # t:29/472
#     axpy!(-1, dy, tmp)                   # t:24/472
#     vecnorm(tmp)^2/(2*ccount(y))         # t:330/472
# end

# # quadlossback(y::Array, dy::Array, dx::Array=dy; o...)=(nx=ccount(dx); for i=1:length(dx); dx[i] = (y[i]-dy[i])/nx; end; dx)
# # @gpu quadlossback(y::CudaArray, dy::CudaArray, dx::CudaArray=dy; o...)=(dx===dy||copy!(dx,dy); cudnnTransformTensor(1/ccount(y), y, -1/ccount(y), dx); dx)  ## cudnnTransformTensor is buggy

# quadlossback(y, dy, dx=dy; o...)=(dx===dy||copy!(dx,dy); scale!(-1/ccount(y), dx); axpy!(1/ccount(y), y, dx); dx)

### LOGPLOSS:

# Cross entropy loss to use after the Logp layer.
# l.y should be normalized log probabilities output by the model.
# p has normalized probabilities from the answer key.
# Normalization is across the last dimension, i.e. sum(p[:,...,:,i])==1
# Overwrites p with the gradient of the loss wrt y, i.e. exp(y)-p:
#
# dy = sum(exp(y))   ;; normalization constant (should be 1 here)
# q = exp(y)/dy      ;; model probabilities
# logq = y - logz   ;; model (normalized) log prob
# dlogz/dy = q      
#
# J = (1/N) Σ[nc] -p[nc]*logq[nc]  ;; n=1..N: instance, c=1..C: class
#   = (1/N) Σ[nc] -p[nc]*(y[nc]-logz[n])
#   = (1/N) ((Σ[n] logz[n]) - (Σ[nc] p[nc]*y[nc]))
#   = (1/N) (Σ[nc] -p[nc]*y[nc])   ;; all logz are 0
#
# dJ/dy[md] = (1/N) (q[md] - p[md])

logplossloss(y::Array, dy::Array)=(nx = ccount(dy); cost = zero(Float64); for i=1:length(dy); cost -= (dy[i]*y[i]); end; cost/nx)
logplossback(y::Array, dy::Array, dx::Array=dy)=(nx = ccount(dx); for i=1:length(dx); dx[i] = (exp(y[i])-dy[i])/nx; end; dx)
@gpu (logplossback(y::CudaArray{Float32}, dy::CudaArray{Float32}, dx::CudaArray{Float32}=dy)=
        (ccall((:logplossback32,libknet),Void,(Cint,Cdouble,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),
               length(dy),1/ccount(dy),y,dy,dx); dx))
@gpu (logplossback(y::CudaArray{Float64}, dy::CudaArray{Float64}, dx::CudaArray{Float64}=dy)=
        (ccall((:logplossback64,libknet),Void,(Cint,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),
               length(dy),1/ccount(dy),y,dy,dx); dx))


### XENTLOSS:

# Cross entropy loss to use after an unnormalized layer.
# l.y is treated as unnormalized log probabilities output by the model.
# p has normalized probabilities from the answer key.
# Normalization is across the last dimension, i.e. sum(p[:,...,:,i])==1
# Overwrites p with the gradient of the loss wrt y, i.e. q-p:
#
# z = sum(exp(y))    ;; normalization constant
# q = exp(y)/z       ;; model probabilities
# logq = y - logz    ;; model (normalized) log prob
# dlogz/dy = q      
#
# J = (1/N) Σ[nc] -p[nc]*logq[nc]  ;; n=1..N: instance, c=1..C: class
#   = (1/N) Σ[nc] -p[nc]*(y[nc]-logz[n])
#   = (1/N) ((Σ[n] logz[n]) - (Σ[nc] p[nc]*y[nc]))
#
# dJ/dy[md] = (1/N) (q[md] - p[md])

function xentlossloss(y::Array, p::Array)
    cost = zero(Float64)
    (nd,nx) = size2(p)
    for j=1:nx
        i1=(j-1)*nd+1; i2=j*nd
        z = sumpy = zero(Float64)
        ymax = typemin(eltype(y))
        for i=i1:i2; y[i] > ymax && (ymax = y[i]); end
        for i=i1:i2; yi=y[i]-ymax; z += exp(yi); sumpy += p[i]*yi; end
        cost += (log(z) - sumpy)
    end
    return cost/nx
end

function xentlossback(y::Array, p::Array, dx::Array=p)
    (nd,nx) = size2(p)
    for j=1:nx
        i1=(j-1)*nd+1; i2=j*nd
        z = zero(Float64)
        ymax = typemin(eltype(y)) # subtract ymax for numerical stability
        for i=i1:i2; y[i] > ymax && (ymax = y[i]); end
        for i=i1:i2; z += exp(y[i]-ymax); end
        for i=i1:i2; yi = exp(y[i]-ymax)/z; dx[i] = (yi - p[i])/nx; end
    end
    return dx
end

@gpu (xentlossback(y::CudaArray{Float32}, p::CudaArray{Float32}, dx::CudaArray{Float32}=p)=
        ((nd,nx)=size2(p);ccall((:xentlossback32,libknet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),nd,nx,y,p,dx);dx))
@gpu (xentlossback(y::CudaArray{Float64}, p::CudaArray{Float64}, dx::CudaArray{Float64}=p)=
        ((nd,nx)=size2(p);ccall((:xentlossback64,libknet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),nd,nx,y,p,dx);dx))


### PERCLOSS

# Perceptron loss function.

# Going forward perceptron computes y=w*x and PercLoss simply records
# the output y.  size(w)=(nc,nd) where nc is the number of classes and
# nd is the number of x dimensions (i.e. features).  size(x)=(nd,nx)
# where nd is the number of features and nx is the batch size.  This
# gives us size(y)=(nc,nx) where the highest entry in each column of y
# indicates the predicted class.

# Going back we get a dy matrix with size(dy)=(nc,nx) where the correct
# answer is marked with the maximum entry in each column.
# For a given column with input x, if cz is the correct answer and cy
# is the predicted answer, the multiclass perceptron update rule is:

# w[cz,:] += x;  w[cy,:] -= x

# Note that there is no update if cz==cy.

# The mmul updates are:
# dw = dy*x'
# dx = w'*dy

# So the perceptron update will be performed if we pass a dy matrix
# back where in each column we have all zeros if the predicted answer
# is correct, otherwise the correct answer is marked with -1 and the
# predicted answer is marked with a +1.  The signs might be confusing,
# this is the gradient of the loss, i.e. going in this direction will
# increase the loss.  We will overwrite the dy matrix.

# This update can be seen as the gradient of a perceptron loss
# function Sum(-y[I]+y[J]) where I are the indices for the correct
# answers, and J are the indices for predicted answers.

function percloss{T}(ypred::Array{T}, ygold::Array{T})
    (nc,nx) = size2(ypred)
    cost = zero(Float64)
    for j=1:nx
        (cz,cy,ymax,zmax) = (0,0,typemin(T),typemin(T))
        i1=(j-1)*nc+1; i2=j*nc
        for i=i1:i2
            ypred[i] > ymax && ((cy,ymax) = (i,ypred[i]))
            ygold[i] > zmax && ((cz,zmax) = (i,ygold[i]))
        end
        (cz != cy) && (cost += ypred[cy]; cost -= ypred[cz])
    end
    return cost/nx
end

function perclossback{T}(ypred::Array{T}, ygold::Array{T}, dx::Array{T}=ygold)
    (nc,nx) = size2(ypred)
    for j=1:nx
        (cz,cy,ymax,zmax) = (0,0,typemin(T),typemin(T))
        i1=(j-1)*nc+1; i2=j*nc
        for i=i1:i2
            ypred[i] > ymax && ((cy,ymax) = (i,ypred[i]))
            ygold[i] > zmax && ((cz,zmax) = (i,ygold[i]))
            dx[i] = zero(T)
        end
        # TODO: these should be scaled 1/nx, why isn't our gradient check complaining?
        (cz != cy) && (dx[cz] = -1; dx[cy] = 1)
    end
    return dx
end

@gpu (perclossback(ypred::CudaArray{Float32}, ygold::CudaArray{Float32}, dx::CudaArray{Float32}=ygold)=((nd,nx)=size2(ygold);ccall((:perclossback32,libknet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),nd,nx,ypred,ygold,dx);dx))
@gpu (perclossback(ypred::CudaArray{Float64}, ygold::CudaArray{Float64}, dx::CudaArray{Float64}=ygold)=((nd,nx)=size2(ygold);ccall((:perclossback64,libknet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),nd,nx,ypred,ygold,dx);dx))


### SCALLOSS
#
# When we do structured training, gradients rather than answers come
# back.  We should scale them using training batch size so the
# learning rate is independent of batch size.  TODO: find a better
# interface.  can't pass back target probabilities because network
# output is not locally normalized.  can we pass back anything so one
# of the existing loss functions would work?

scallossloss(y,dy)=error("Not implemented")
scallossback(y,dy,dx=dy)=(dx===dy||copy!(dx,dy);scale!(1/ccount(dx), dx))


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

### DEAD CODE:

        # TODO: can we take these out and make them apply to Loss?
        # $lloss(y::KUdense{Array}, dy::KUdense{Array}; o...)=$lloss(convert(Array,y), convert(Array,dy); o...)
        # $lloss(y::KUdense{Array}, dy::Array; o...)=$lloss(convert(Array,y), convert(Array,dy); o...)
        # $lloss(y::KUdense{Array}, dy::KUdense{Array}; o...)=$lloss(convert(Array,y), convert(Array,dy); o...)
        # $lloss(y::KUdense{Array}, dy::SparseMatrixCSC; o...)=$lloss(convert(Array,y), dy; o...)
        # @gpu $lloss{T}(y::KUdense{CudaArray,T},dy::Array{T}; o...)=$lloss(convert(CudaArray,y), convert(CudaArray,dy); o...)
        # @gpu $lloss{T}(y::KUdense{CudaArray,T},dy::KUdense{Array,T}; o...)=$lloss(convert(CudaArray,y), convert(CudaArray,dy); o...)
        # @gpu $lloss{T}(y::KUdense{CudaArray,T},dy::SparseMatrixCSC{T}; o...)=$lloss(convert(CudaArray,y), convert(CudaSparseMatrixCSC,dy); o...)

        # $lback(y::KUdense{Array}, dy::KUdense{Array}, dx::KUdense{Array};o...)=($lback(convert(Array,y), convert(Array,dy), convert(Array, dx);o...); dx)
        # $lback(y::KUdense{Array}, dy::SparseMatrixCSC, dx::KUdense{Array};o...)=($lback(convert(Array,y), dy, convert(Array, dx);o...); dx)
        # @gpu $lback(y::KUdense{CudaArray}, dy::KUdense{CudaArray}, dx::KUdense{CudaArray};o...)=($lback(convert(CudaArray,y), convert(CudaArray,dy), convert(CudaArray, dx);o...); dx)
        # @gpu $lback(y::KUdense{CudaArray}, dy::CudaSparseMatrixCSC, dx::KUdense{CudaArray};o...)=($lback(convert(CudaArray,y), dy, convert(CudaArray, dx);o...); dx)

        # $lback(y::KUdense, dy::KUdense, dx::KUdense=dy)=($lback(y.arr, dy.arr, dx.arr); dx)
        # $lloss(y,dy)=$lloss(convert(Array,y), convert(Array,dy))  # TODO: handle sparse arrays, implement gpu

# params(::Loss)=Any[]
# overwrites(::Loss)=true
