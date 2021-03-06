"""
    reshape2d(x; dims = ndims(x) - 1)

Reshape `x` into a two-dimensional matrix by joining the first dims dimensions, i.e. 
`reshape(x, prod(size(x,i) for i in 1:dims), :)`

`dims=ndims(x)-1` (default) is typically used when turning the output of a 4-D convolution
result into a 2-D input for a fully connected layer.

`dims=1` is typically used when turning the 3-D output of an RNN layer into a 2-D input for
a fully connected layer.

`dims=0` will turn the input into a row vector, `dims=ndims(x)` will turn it into a column
vector.

"""
reshape2d(x; dims=ndims(x)-1)=reshape(x, (dims > 0 ? prod(size(x,i) for i in 1:dims) : 1), :)
