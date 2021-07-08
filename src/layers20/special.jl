"""
    MLP(h::Int...;kwargs...)


Creates a multi layer perceptron according to given `h`s.
First `h` should be input size and the last one should be output size.

    (m::MLP)(x)

Runs MLP with given input `x`.

# Keywords

* `winit=xavier`: weight initialization distribution
* `bias=zeros`: bias initialization distribution
* `activation=ReLU()`: activation layer between layers
* `atype=KnetLayers.arrtype` : array type for parameters.
   Default value is KnetArray{Float32} if you have gpu device. Otherwise it is Array{Float32}
*  pdrop=0: dropout probability between layers


"""
mutable struct MLP <: Layer
    layers::Tuple{Vararg{Linear}}
    activation::Activation
    drop::Dropout
end

function MLP(h::Int...; winit=xavier, binit=zeros, activation=ReLU(), atype=arrtype, pdrop=0)
    singlelayer(input,ouput) = Linear(input=input, output=ouput, winit=winit, binit=binit, atype=atype)
    MLP(singlelayer.(h[1:end-1], h[2:end]),activation,Dropout(pdrop))
end

function (m::MLP)(x)
    for layer in m.layers
        x = layer(m.drop(x))
        if layer !== m.layers[end]
            x = m.activation(x)
        end
    end
    return x
end
