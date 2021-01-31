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
    sshow(io, r.f1, indent+2)
    sshow(io, '+', indent+2)
    sshow(io, r.f2, indent+2)
    #sshow(io, "end", indent)
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
show(io::IO, b::BatchNorm) = print(io, "BatchNorm()")


show(io::IO, ::MIME"text/plain", c::Conv) = show(io, c)
function show(io::IO, c::Conv)
    @printf(io, "Conv(%d×%d, %d=>%d", c.wdims...)
    # Only print non-defaults
    if c.w !== nothing && eltype(c.w) !== Float32; print(io, ", ", eltype(c.w)); end
    if c.padding != 0; print(io, ", padding=$(c.padding)"); end
    if c.stride != 1; print(io, ", stride=$(c.stride)"); end
    if c.dilation != 1; print(io, ", dilation=$(c.dilation)"); end
    if c.groups != 1; print(io, ", groups=$(c.groups)"); end
    if c.crosscorrelation; print(io, ", crosscorrelation"); end
    if c.channelmajor; print(io, ", channelmajor"); end
    if c.alpha != 1; print(io, ", alpha=$(c.alpha)"); end
    if c.beta != 0; print(io, ", beta=$(c.beta)"); end
    if c.bias !== nothing; print(io, ", bias"); end
    if c.normalization !== nothing; print(io, ", $(c.normalization)"); end
    if c.activation !== nothing; print(io, ", $(c.activation)"); end
    print(io, ")")
end


show(io::IO, ::MIME"text/plain", d::Dense) = show(io, d)
function show(io::IO, d::Dense)
    print(io, "Dense($(d.inputsize)=>$(d.outputsize)")
    if d.w !== nothing && eltype(d.w) !== Float32; print(io, ", ", eltype(d.w)); end
    if d.bias !== nothing; print(io, ", bias"); end
    if d.activation !== nothing; print(io, ", $(d.activation)"); end
    if d.dropout !== 0; print(io, ", dropout=$(d.dropout)"); end
    print(io, ")")
end
