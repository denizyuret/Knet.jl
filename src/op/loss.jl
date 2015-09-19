# Loss Layers
# TODO: rename LossLayer -> Loss

abstract LossLayer <: Op

params(::LossLayer)=Any[]
ninputs(::LossLayer)=1
ysize(::LossLayer,x)=size(x)
overwrites(::LossLayer)=true
back_reads_x(::LossLayer)=false
back_reads_y(::LossLayer)=true

# LossLayer has slightly different input/output behavior compared to regular layers:
# forw only records the outgoing y.
# back takes dy, the desired output, and overwrites it with the loss gradient wrt y
# loss takes dy, the desired output, and returns a loss value

for (ltype, lback, lloss) in (
                              (:QuadLoss, :quadlossback, :quadloss),
                              (:SoftLoss, :softlossback, :softloss),
                              (:LogpLoss, :logplossback, :logploss),
                              (:XentLoss, :xentlossback, :xentloss),
                              (:PercLoss, :perclossback, :percloss),
                              (:ScalLoss, :scallossback, :scalloss),
                              )
    @eval begin
        type $ltype <: LossLayer; y; $ltype()=new(); end
        $lback(y::KUdense, dy::KUdense, dx::KUdense=dy)=($lback(y.arr, dy.arr, dx.arr); dx)
        $lloss(y,dy)=$lloss(convert(Array,y), convert(Array,dy))  # TODO: handle sparse arrays
        loss(l::$ltype, dy; y=l.y, o...)=$lloss(y,dy) # (issimilar(dy,y)||error(map(summary,(y,dy))); $lloss(y,dy))
        forw(l::$ltype, x; y=x, o...)=(issimilar(x,y)||error("x/y"); l.y = (y===x ? y : copy!(y,x)))
        back(l::$ltype, dy; dx=dy, y=l.y, returndx=true, o...)=
            (returndx||return; (issimilar(dy,y) && issimilar(dx,y))||error("$y\n$dy\n$dx"); $lback(y,dy,dx))
    end
end

### QUADLOSS:

# Quadratic loss:
# l.y stores the model output.
# dy is the desired output.
# Overwrites dy with the gradient of quadratic loss wrt y, i.e. y-dy
# J = 0.5*sum((yi-zi)^2)
# dJ/dy = y-dy

quadloss(y::Array, dy::Array)=(cost=zero(Float64); for i=1:length(dy); cost += (y[i]-dy[i])^2; end; 0.5*cost/ccount(dy))
quadlossback(y::Array, dy::Array, dx::Array=dy)=(nx=ccount(dx); for i=1:length(dx); dx[i] = (y[i]-dy[i])/nx; end; dx)
GPU && (quadlossback(y::CudaArray, dy::CudaArray, dx::CudaArray=dy)=
        (dx===dy||copy!(dx,dy); cudnnTransformTensor(1/ccount(y), y, -1/ccount(y), dx); dx))


### SOFTLOSS: 

# Cross entropy loss to use after the Soft layer.
# l.y should have normalized probabilities output by the model.
# p has normalized probabilities from the answer key.
# Normalization is across the last dimension, i.e. sum(p[:,...,:,i])==1
# Overwrites p with the gradient of the loss wrt y, i.e. 1-p/y

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

softloss(y::Array,dy::Array)=(cost=zero(Float64); for i=1:length(dy); dy[i]>0 && (cost -= (dy[i]*log(y[i]))); end; cost/ccount(dy))
softlossback(y::Array,dy::Array,dx::Array=dy)=(nx=ccount(dx); for i=1:length(dx); dx[i] = ((y[i]-dy[i])/y[i])/nx; end; dx)
GPU && (softlossback(y::CudaArray{Float32}, dy::CudaArray{Float32}, dx::CudaArray{Float32}=dy)=(ccall((:softloss32,libkunet),Void,(Cint,Cdouble,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),length(dy),1/ccount(dy),y,dy,dx);dx))
GPU && (softlossback(y::CudaArray{Float64}, dy::CudaArray{Float64}, dx::CudaArray{Float64}=dy)=(ccall((:softloss64,libkunet),Void,(Cint,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),length(dy),1/ccount(dy),y,dy,dx);dx))


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

logploss(y::Array, dy::Array)=(nx = ccount(dy); cost = zero(Float64); for i=1:length(dy); cost -= (dy[i]*y[i]); end; cost/nx)
logplossback(y::Array, dy::Array, dx::Array=dy)=(nx = ccount(dx); for i=1:length(dx); dx[i] = (exp(y[i])-dy[i])/nx; end; dx)
GPU && (logplossback(y::CudaArray{Float32}, dy::CudaArray{Float32}, dx::CudaArray{Float32}=dy)=
        (ccall((:logploss32,libkunet),Void,(Cint,Cdouble,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),
               length(dy),1/ccount(dy),y,dy,dx); dx))
GPU && (logplossback(y::CudaArray{Float64}, dy::CudaArray{Float64}, dx::CudaArray{Float64}=dy)=
        (ccall((:logploss64,libkunet),Void,(Cint,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),
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

function xentloss(y::Array, p::Array)
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

GPU && (xentlossback(y::CudaArray{Float32}, p::CudaArray{Float32}, dx::CudaArray{Float32}=p)=
        ((nd,nx)=size2(p);ccall((:xentloss32,libkunet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),nd,nx,y,p,dx);dx))
GPU && (xentlossback(y::CudaArray{Float64}, p::CudaArray{Float64}, dx::CudaArray{Float64}=p)=
        ((nd,nx)=size2(p);ccall((:xentloss64,libkunet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),nd,nx,y,p,dx);dx))


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

function percloss{T}(y::Array{T}, dy::Array{T})
    (nc,nx) = size2(y)
    cost = zero(Float64)
    for j=1:nx
        (cz,cy,ymax,zmax) = (0,0,typemin(T),typemin(T))
        i1=(j-1)*nc+1; i2=j*nc
        for i=i1:i2
            y[i] > ymax && ((cy,ymax) = (i,y[i]))
            dy[i] > zmax && ((cz,zmax) = (i,dy[i]))
        end
        (cz != cy) && (cost += y[cy]; cost -= y[cz])
    end
    return cost/nx
end

function perclossback{T}(y::Array{T}, dy::Array{T}, dx::Array{T}=dy)
    (nc,nx) = size2(y)
    for j=1:nx
        (cz,cy,ymax,zmax) = (0,0,typemin(T),typemin(T))
        i1=(j-1)*nc+1; i2=j*nc
        for i=i1:i2
            y[i] > ymax && ((cy,ymax) = (i,y[i]))
            dy[i] > zmax && ((cz,zmax) = (i,dy[i]))
            dx[i] = zero(T)
        end
        # TODO: these should be scaled 1/nx, why isn't our gradient check complaining?
        (cz != cy) && (dx[cz] = -1; dx[cy] = 1)
    end
    return dx
end

GPU && (perclossback(y::CudaArray{Float32}, dy::CudaArray{Float32}, dx::CudaArray{Float32}=dy)=((nd,nx)=size2(dy);ccall((:percloss32,libkunet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),nd,nx,y,dy,dx);dx))
GPU && (perclossback(y::CudaArray{Float64}, dy::CudaArray{Float64}, dx::CudaArray{Float64}=dy)=((nd,nx)=size2(dy);ccall((:percloss64,libkunet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),nd,nx,y,dy,dx);dx))


### SCALLOSS
#
# When we do structured training, gradients rather than answers come
# back.  We should scale them using training batch size so the
# learning rate is independent of batch size.  TODO: find a better
# interface.  can't pass back target probabilities because network
# output is not locally normalized.  can we pass back anything so one
# of the existing loss functions would work?

scalloss(y,dy)=error("Not implemented")
scallossback(y,dy,dx=dy)=(dx===dy||copy!(dx,dy);scale!(1/ccount(dx), dx))
