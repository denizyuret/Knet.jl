import Base: show
using Printf

# show(io::IO, x): One line show used for show, print, string etc.
# show(io::IO, ::MIME"text/plain", x): Multi line show used for display

# show with indent:
sshow(io::IO, x, indent) = print(io, repeat(' ', indent), x, '\n')

show(io::IO, ::MIME"text/plain", s::Sequential) = show(io, s)
show(io::IO, s::Sequential) = sshow(io, s, 0)
function sshow(io::IO, s::Sequential, indent::Int)
    sshow(io, (s.name === nothing ? "Sequential" : "$(s.name)"), indent)
    for l in s.layers
        sshow(io, l, indent+2)
    end
    #sshow(io, "end", indent)
end


show(io::IO, ::MIME"text/plain", s::Residual) = show(io, s)
show(io::IO, s::Residual) = sshow(io, s, 0)
function sshow(io::IO, r::Residual, indent::Int)
    a = r.activation === nothing ? "" : "($(r.activation))"
    sshow(io, "Residual$a", indent)
    sshow(io, r.blocks[1], indent+2)
    for i in 2:length(r.blocks)
        if r.blocks[i] == identity
            sshow(io, ".+ identity", indent+2)
        else
            sshow(io, ".+", indent+2)
            sshow(io, r.blocks[i], indent+2)
        end
    end
end


show(io::IO, ::MIME"text/plain", s::SqueezeExcitation) = show(io, s)
show(io::IO, s::SqueezeExcitation) = sshow(io, s, 0)
function sshow(io::IO, r::SqueezeExcitation, indent::Int)
    sshow(io, "SqueezeExcitation", indent)
    sshow(io, r.block, indent+2)
    sshow(io, ".* identity", indent+2)
end


function sshow(io::IO, f::Function, indent::Int)
    name = string(f)
    if name[1] === '#'          # handle anonymous functions
        m = first(methods(f))
        lines = readlines(string(m.file))
        src = strip(lines[m.line], [' ', '\t', ',', ';'])
        sshow(io, src, indent)
    else
        sshow(io, name, indent)
    end
end


show(io::IO, ::MIME"text/plain", b::BatchNorm) = show(io, b)
show(io::IO, b::BatchNorm) = print(io, BatchNorm)


show(io::IO, ::MIME"text/plain", c::Conv) = show(io, c)
function show(io::IO, c::Conv)
    @printf(io, "Conv(%d×%d, %d⇒%d", c.wdims...)
    # Only print non-defaults
    if c.w !== nothing && eltype(c.w) !== Float32; print(io, ", ", eltype(c.w)); end
    if any(c.padding .!= 0); print(io, ", padding=$(c.padding)"); end
    if any(c.stride .!= 1); print(io, ", stride=$(c.stride)"); end
    if any(c.dilation .!= 1); print(io, ", dilation=$(c.dilation)"); end
    if c.groups != 1; print(io, ", groups=$(c.groups)"); end
    if !c.crosscorrelation; print(io, ", flipkernel"); end
    if c.channelmajor; print(io, ", channelmajor"); end
    if c.alpha != 1; print(io, ", alpha=$(c.alpha)"); end
    if c.beta != 0; print(io, ", beta=$(c.beta)"); end
    if c.bias !== nothing; print(io, ", bias"); end
    if c.normalization !== nothing; print(io, ", $(c.normalization)"); end
    if c.activation !== nothing; print(io, ", $(c.activation)"); end
    print(io, ")")
end


show(io::IO, ::MIME"text/plain", d::Linear) = show(io, d)
function show(io::IO, d::Linear)
    print(io, "Linear($(d.inputsize)=>$(d.outputsize)")
    if d.w !== nothing && eltype(d.w) !== Float32; print(io, ", ", eltype(d.w)); end
    if d.bias !== nothing; print(io, ", bias"); end
    if d.activation !== nothing; print(io, ", $(d.activation)"); end
    if d.dropout !== 0; print(io, ", dropout=$(d.dropout)"); end
    print(io, ")")
end


show(io::IO, ::MIME"text/plain", o::Op) = show(io, o)
function show(io::IO, o::Op)
    print(io, o.f)
    !isempty(o.args) && print(io, o.args)
    !isempty(o.kwargs) && print(io, (; o.kwargs...))
end


show(io::IO, ::MIME"text/plain", z::ZeroPad) = show(io, z)
show(io::IO, z::ZeroPad) = print(io, "ZeroPad$(z.padding)")
