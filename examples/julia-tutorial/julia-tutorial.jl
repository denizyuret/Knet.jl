# Julia tutorial. Concepts:
# - Pkg: clone, add, rm, update, init, status, using
# - Arrays: ranges, tuples, multidimensional arrays, indexing, eltype, size, length, sizeof, ndims, transpose, mean, std, minimum, maximum, reshape, vec, randn, rand, zeros, ones, unary ops, element-wise ops, broadcasting ops, reductions, sub, map, copy, copy!, == vs ===, @time, CudaArray, concatenation
# - Functions: if, for, function, anonymous function, map, optional args, keyword args, three-dot notation, Dict, Symbol

# Packages
julia> Pkg.init()
julia> Pkg.add("Knet")
julia> Pkg.status()
julia> Pkg.add("ASCIIPlots")
julia> using ASCIIPlots
julia> a = eye(3)
julia> imagesc(a)
julia> Pkg.rm("ASCIIPlots")
julia> Pkg.update()


# Arrays: indexing
julia> a=reshape([1:15;], (3,5))
julia> a[1,1]
julia> a[3,5]
julia> a[4]

# Ranges
julia> 1:10
julia> for i=1:10; println(i); end
julia> for i=1:3:10; println(i); end
julia> 1:100000000000
julia> [1:10;]
julia> a=1:10
julia> typeof(a)
julia> a=1:0.5:10
julia> typeof(a)
julia> [a;]
julia> collect(a)

# Indexing continued
julia> a[3,1:2]
julia> a[2:3,1:2]
julia> a
julia> a[2:3,:]
julia> b=[1,3,5]
julia> a[:,b]
julia> b=[false, true, false, true, false]
julia> a[:,b]
julia> a
julia> a[10]
julia> a
julia> a[1]
julia> a[3:5]
julia> a = randn(10)
julia> a[a.>0]
julia> a.>0
julia> a
julia> b = (a.>0)
julia> a[b]

# Arrays: basic ops
julia> a = zeros(3,5)
julia> eltype(a)
julia> size(a)
julia> size(a,1)
julia> size(a,2)
julia> length(a)
julia> sizeof(a)
julia> ndims(a)
julia> transpose(a)
julia> a.'
julia> a'

# Arrays: unary ops
julia> a = rand(3,5)
julia> sin(a)
julia> -a
julia> 10a

# Arrays: binary ops
julia> a = rand(3,5)
julia> b = rand(3,5)
julia> a + b
julia> a * b  => matmul
julia> a .* b => element-wise
julia> a
julia> b
julia> a .+ b
julia> a + b
julia> b = rand(5,2)
julia> a * b
julia> a = rand(2000,1000)
julia> b = rand(1000, 3000)
julia> c = a*b
julia> a = rand(10000, 2000)
julia> b = rand(2000, 5000)
julia> c = a * b
julia> map(size, (a,b,c))

# Arrays: broadcasting
julia> a = rand(3,5)
julia> a > 0.5
julia> a .> 0.5
julia> b = rand(3,1)+10
julia> a + b
julia> a .+ b
julia> a
julia> b
julia> broadcast(+, a, b)
julia> a + b

# Arrays: reductions
julia> a = rand(3,5)
julia> mean(a)
julia> maximum(a)
julia> minimum(a)
julia> mean(a)
julia> mean(a,1)
julia> mean(a,2)
julia> minimum(a,1)

# Arrays: slices vs subarrays
julia> a = rand(3,5)
julia> b = a[1:2,1:2]
julia> b[1,1]=0
julia> b
julia> a
julia> b = sub(a, 1:2, 1:2)
julia> a
julia> b
julia> b[1,1]=0
julia> b
julia> a

# Arrays: concatenation
julia> a = rand(3,3)
julia> b = rand(3)
julia> hcat(a,b)
julia> [a b]
julia> vcat(a,b) => Error
julia> vcat(a,b')
julia> [a; b']

# Arrays: reshaping
julia> a = reshape([1:15.0;], 3, 5)
julia> vec(a)
julia> a[1,2]
julia> a[2,2]
julia> a[2,3]
julia> a[2,4]
julia> a
julia> reshape(a, (5,3))
julia> a
julia> size(a,1)
julia> a

# Arrays: copying and identity
julia> a
julia> d = copy(a)
julia> d[1]=0
julia> d
julia> a
julia> a===d
julia> copy!(d, a)
julia> d
julia> a
julia> a==d
julia> a===d
julia> f = a
julia> a==f
julia> a===f
julia> a==d
julia> a===d

# KnetArrays: (for GPU machines)
julia> a = rand(10000, 2000)
julia> b = rand(2000, 5000)
julia> @time c = a * b;
julia> @time c = a * b;
julia> @time c = a * b;
julia> @time c = a * b;
julia> using Knet
julia> a2 = KnetArray(a)
julia> b2 = KnetArray(b)
julia> @time c2 = a2 * b2;
julia> c3 = Array(c2)
julia> isapprox(c,c3)

# Conditionals and Loops:
julia> if 1 > 0; println(:foo); else; println(:bar); end
julia> if 1 > 0; println(:foo); elseif 2 > 3; println(:bar); else 0
julia> for i=1:10; println(i); end

# Functions:
julia> function f(x,y); x+y; end
julia> f(2,3)

# Functions and files:
julia> pwd()
julia> cd("tutorial")
julia> include("foo.jl")
julia> foo(4,5)
julia> include("foo.jl")
julia> foo(4,5)
julia> include("foo.jl")
julia> foo(4,5)
julia> include("foo.jl")
julia> foo(4,5)

# Tuples
julia> a=(1,2,3)
julia> typeof(a)
julia> a[1]
julia> a[1] = 10 => Error

# Functions that return tuples:
julia> include("foo.jl")
julia> foo(4,5) => (23,9)
julia> var1,var2 = foo(4,5)
julia> var1
julia> var2

# Functions: anonymous
julia> function (x); x+1; end
julia> x->x+1
julia> a = rand(10)
julia> map(x->2x, a)
julia> bar(x,y)=2x+3y
julia> bar(3,5)
julia> foo(3,5)

# Functions: first class entities
julia> baz = foo
julia> baz(3,5)
julia> baz === foo
julia> baz == foo

# Functions: optional arguments
julia> include("foo.jl")
julia> foo(3,5) => (21,8)
julia> foo(3) => (15,6)
julia> foo() => Error
julia> foo(1,2,3) => Error

# Functions: keyword arguments
julia> include("foo.jl")
julia> foo(1,2)
julia> foo(1,2,scale=100)
julia> foo(1,2;scale=100)
julia> foo(1,2,100) => Error
julia> include("foo.jl")
julia> foo(1,2)
julia> foo(1,2,scale=100)
julia> foo(1,2,bias=30)
julia> foo(1,2,bias=30,scale=200)

# Functions variable number of arguments
julia> f(x,y...)=println("x=$x y=$y")
julia> f()
julia> f(3)
julia> f(3,4)
julia> z(x,y...)=println("x=$x y=$y")
julia> z(3,4)
julia> z(3,4,5,6,7,8)
julia> q(x,y; scale=10, kwargs...)=println("x=$x y=$y scale=$scale kwargs=$kwargs")
julia> q(1,2)
julia> q(1,2; scale=3)
julia> q(1,2; fish=6)
julia> q(1,2; fish=6, dog=1)

# Dictionaries
julia> d=Dict()
julia> d[:foo]=1
julia> d[:bar]=2
julia> d
julia> d[:bar]
julia> d[:baz]

# Dictionaries as keyword argument lists
julia> q(1,2; d...)
julia> q2(x,y; scale=10, kwargs...)=println("x=$x y=$y scale=$scale kwargs[foo]=$(Dict(kwargs)[:foo])")
julia> q2(1,2; d...)
