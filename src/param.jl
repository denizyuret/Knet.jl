# UpdateParam: Parameters can be set at the level of weights, layers, nets, or
# training sessions.  Here is the lowest level each parameter can be
# set:
# weight-specific: learningRate; l1reg; l2reg; maxnorm; adagrad; momentum; nesterov; 
# layer-specific: dropout, apply_fx, return_dx
# net-specific: ? 
# train-specific: batch, iters, loss

type UpdateParam learningRate; l1reg; l2reg; maxnorm; adagrad; ada; momentum; mom; nesterov; nes; 
    UpdateParam(;args...)=(o=setparam!(new(); args...); isdefined(o,:learningRate)||(o.learningRate=1f-2); o)
end

setparam!(x; args...)=(for (k,v)=args; setparam!(x,k,v); end; x)
setparam!(p::UpdateParam,k,v)=(p.(k)=convert(Float32, v); p)
setparam!(net::Net,k,v)=(for l=net; setparam!(l,k,v) end; net)

function setparam!(l::AbstractLayer,k,v)
    if in(k, names(l))
        l.(k) = v
        (k == :dropout) && (v > zero(v)) && (l.fx = drop)
    else
        if isdefined(l, :w)
            isdefined(l, :pw) || (l.pw = UpdateParam())
            setparam!(l.pw, k, v)
        end
        if isdefined(l, :b)
            isdefined(l, :pb) || (l.pb = UpdateParam())
            if in(k, [:l1reg, :l2reg, :maxnorm]) && (v != zero(v))
                warn("Skipping $k regularization for bias.")
            else
                setparam!(l.pb, k, v)
            end
        end
    end
    return l
end

