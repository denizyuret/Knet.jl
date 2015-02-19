# TODO: express the math in readable and generic form 

type TrainOpts adagrad; batch; dropout; iters; l1reg; l2reg; learningRate; maxnorm; momentum; nesterov; 
TrainOpts()=new(0f0,    128,   0f0,     0,     0f0,   0f0,   0.01f0,       0f0,     0f0,      0f0) end

function update(l::Layer, o::TrainOpts)
    initupdate(l, o)
    if o.l1reg > 0f0
        l1reg!(o.l1reg, l.w, l.dw)  # l.dw += o.l1reg * sign(l.w)
    end
    if o.l2reg > 0f0
        l2reg!(o.l2reg, l.w, l.dw)  # l.dw += o.l2reg * l.w
    end
    if o.adagrad > 0f0
        adagrad!(o.adagrad, l.dw2, l.dw) # l.dw2 += l.dw .* l.dw; l.dw /= o.adagrad + sqrt(l.dw2)
        adagrad!(o.adagrad, l.db2, l.db) # l.db2 += l.db .* l.db; l.db /= o.adagrad + sqrt(l.db2)
    end
    if o.learningRate != 1.0f0
        @in1! l.dw .* o.learningRate
        @in1! l.db .* o.learningRate
    end
    if o.momentum > 0f0
        assert(o.nesterov == 0f0)
        momentum!(o.momentum, l.dw1, l.dw)  # l.dw1 = o.momentum * l.dw1 + l.dw; l.dw = l.dw1
        momentum!(o.momentum, l.db1, l.db)  # l.db1 = o.momentum * l.db1 + l.db; l.db = l.db1
    end
    if o.nesterov > 0f0
        assert(o.momentum == 0f0)
        nesterov!(o.nesterov, l.dw1, l.dw)  # l.dw1 = o.nesterov * l.dw1 + l.dw; l.dw = o.nesterov * l.dw1 + l.dw; 
        nesterov!(o.nesterov, l.db1, l.db)  # l.db1 = o.nesterov * l.db1 + l.db; l.db = o.nesterov * l.db1 + l.db; 
    end
    @in1! l.w .- l.dw
    @in1! l.b .- l.db
    if o.maxnorm > 0f0
        norms = sqrt(sum(w.^2, 2))
        if any(norms > o.maxnorm)
            scale = min(o.maxnorm ./ norms, 1)
            l.w *= scale
        end
    end
end

function initupdate(l, o)
    if o.adagrad > 0f0
        if (!isdefined(l,:dw2)) l.dw2 = zeros(l.dw) end
        if (!isdefined(l,:db2)) l.db2 = zeros(l.db) end
    end
    if (o.momentum > 0f0 || o.nesterov > 0f0)
        if (!isdefined(l,:dw1)) l.dw1 = zeros(l.dw) end
        if (!isdefined(l,:db1)) l.db1 = zeros(l.db) end
    end
end

l2reg!(l2::Float32, w::Matrix, dw::Matrix) =    Base.LinAlg.axpy!(length(dw), l2, w, 1, dw, 1)
l2reg!(l2::Float32, w::CudaMatrix, dw::CudaMatrix) = CUBLAS.axpy!(length(dw), l2, w, 1, dw, 1)

l1reg!(l1::Float32, w::CudaMatrix, dw::CudaMatrix)=ccall((:l1reg,libkunet),Void,(Cint,Cfloat,Cmat,Cmat),length(dw),l1,w,dw)

function l1reg!(l1::Float32, w::Matrix, dw::Matrix)
    for i=1:length(dw)
        if (w[i] > 0f0) dw[i] += l1
        elseif (w[i] < 0f0) dw[i] -= l1
        end
    end
end

adagrad!(eps::Float32, dw2::CudaMatrix, dw::CudaMatrix)=ccall((:adagrad,libkunet),Void,(Cint,Cfloat,Cmat,Cmat),length(dw),eps,dw2,dw)

function adagrad!(eps::Float32, dw2::Matrix, dw::Matrix)
    for i=1:length(dw)
        dw2[i] += dw[i] * dw[i];
        dw[i] /= (eps + sqrt(dw2[i]))
    end
end

# TODO: use blas for gpu versions to take advantage of multicore.
function momentum!(m::Float32, dw1::Matrix, dw::Matrix)
    for i=1:length(dw)
        dw1[i] = m * dw1[i] + dw[i]
        dw[i]  = dw1[i]
    end
end

function momentum!(m::Float32, dw1::CudaMatrix, dw::CudaMatrix)
    CUBLAS.axpy!(length(dw), m, dw1, 1, dw, 1)
    copy!(dw1, dw)
end

function nesterov!(m::Float32, dw1::Matrix, dw::Matrix)
    for i=1:length(dw)
        dw1[i] = m * dw1[i] + dw[i]
        dw[i]  = m * dw1[i] + dw[i]
    end
end

function nesterov!(m::Float32, dw1::CudaMatrix, dw::CudaMatrix)
    CUBLAS.scal!(length(dw), m, dw1, 1)
    CUBLAS.axpy!(length(dw), 1.0f0, dw, 1, dw1, 1)
    CUBLAS.axpy!(length(dw), m, dw1, 1, dw, 1)
end
