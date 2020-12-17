# The functions with underscore in the name are manual implementations, the others are cudnn calls.

using CUDA, NNlib, BenchmarkTools, Knet, Test
using NNlib: softmax, softmax!, ∇softmax, ∇softmax!, logsoftmax, logsoftmax!, ∇logsoftmax, ∇logsoftmax!
using CUDA.CUDNN: _softmax!, _∇softmax!, _logsoftmax!, _∇logsoftmax!

x,y,dx,dy = (CUDA.randn(1024, 256) for i in 1:4)

macro ctime(ex)
    quote
        println($(sprint(Base.show_unquoted,ex)))
        @btime (CUDA.@sync $ex; nothing)
        CUDA.@time $ex
    end |> esc
end

@test softmax!(similar(y),x;dims=1) ≈ _softmax!(similar(y),x;dims=1)
@test softmax!(similar(y),x;dims=2) ≈ _softmax!(similar(y),x;dims=2)
@test softmax!(similar(y),x;dims=:) ≈ _softmax!(similar(y),x;dims=:)
@test logsoftmax!(similar(y),x;dims=1) ≈ _logsoftmax!(similar(y),x;dims=1)
@test logsoftmax!(similar(y),x;dims=2) ≈ _logsoftmax!(similar(y),x;dims=2)
@test logsoftmax!(similar(y),x;dims=:) ≈ _logsoftmax!(similar(y),x;dims=:)
@test ∇softmax!(similar(dx),dy,x,softmax(x;dims=1);dims=1) ≈ _∇softmax!(similar(dx),dy,x,softmax(x;dims=1);dims=1)
@test ∇softmax!(similar(dx),dy,x,softmax(x;dims=2);dims=2) ≈ _∇softmax!(similar(dx),dy,x,softmax(x;dims=2);dims=2)
@test ∇softmax!(similar(dx),dy,x,softmax(x;dims=:);dims=:) ≈ _∇softmax!(similar(dx),dy,x,softmax(x;dims=:);dims=:)
@test ∇logsoftmax!(similar(dx),dy,x,logsoftmax(x;dims=1);dims=1) ≈ _∇logsoftmax!(similar(dx),dy,x,logsoftmax(x;dims=1);dims=1)
@test ∇logsoftmax!(similar(dx),dy,x,logsoftmax(x;dims=2);dims=2) ≈ _∇logsoftmax!(similar(dx),dy,x,logsoftmax(x;dims=2);dims=2)
@test ∇logsoftmax!(similar(dx),dy,x,logsoftmax(x;dims=:);dims=:) ≈ _∇logsoftmax!(similar(dx),dy,x,logsoftmax(x;dims=:);dims=:)

GC.gc(true)
if true
@ctime softmax!(y,x;dims=1)
@ctime _softmax!(y,x;dims=1)
@ctime softmax!(y,x;dims=2)
@ctime _softmax!(y,x;dims=2)
@ctime softmax!(y,x;dims=:)
@ctime _softmax!(y,x;dims=:)
end

#=
softmax!(y, x; dims = 1)
  29.702 μs (24 allocations: 624 bytes)
  0.015599 seconds (7.92 k CPU allocations: 438.803 KiB)
_softmax!(y, x; dims = 1)
  43.810 μs (51 allocations: 1.41 KiB)
  0.000085 seconds (60 CPU allocations: 1.750 KiB) (2 GPU allocations: 2.000 KiB, 2.58% gc time)
softmax!(y, x; dims = 2)
  158.163 μs (24 allocations: 624 bytes)
  0.000186 seconds (33 CPU allocations: 976 bytes)
_softmax!(y, x; dims = 2)
  69.622 μs (59 allocations: 1.56 KiB)
  0.000104 seconds (68 CPU allocations: 1.906 KiB) (2 GPU allocations: 8.000 KiB, 2.52% gc time)
softmax!(y, x; dims = (:))
  399.958 μs (24 allocations: 624 bytes)
  0.000498 seconds (31 CPU allocations: 944 bytes)
_softmax!(y, x; dims = (:))
  83.624 μs (37 allocations: 1.00 KiB)
  0.000127 seconds (44 CPU allocations: 1.312 KiB)
=#

GC.gc(true)
if true
@ctime logsoftmax!(y,x;dims=1)
@ctime _logsoftmax!(y,x;dims=1)
@ctime logsoftmax!(y,x;dims=2)
@ctime _logsoftmax!(y,x;dims=2)
@ctime logsoftmax!(y,x;dims=:)
@ctime _logsoftmax!(y,x;dims=:)
end

#=
logsoftmax!(y, x; dims = 1)
  30.771 μs (24 allocations: 624 bytes)
  0.000073 seconds (33 CPU allocations: 976 bytes)
_logsoftmax!(y, x; dims = 1)
  60.873 μs (71 allocations: 1.97 KiB)
  0.000126 seconds (80 CPU allocations: 2.312 KiB) (3 GPU allocations: 1.002 MiB, 3.46% gc time)
logsoftmax!(y, x; dims = 2)
  130.960 μs (24 allocations: 624 bytes)
  0.000202 seconds (33 CPU allocations: 976 bytes)
_logsoftmax!(y, x; dims = 2)
  78.876 μs (79 allocations: 2.12 KiB)
  0.000107 seconds (88 CPU allocations: 2.469 KiB) (3 GPU allocations: 1.008 MiB, 3.27% gc time)
logsoftmax!(y, x; dims = (:))
  349.391 μs (24 allocations: 624 bytes)
  0.000366 seconds (31 CPU allocations: 944 bytes)
_logsoftmax!(y, x; dims = (:))
  136.404 μs (57 allocations: 1.56 KiB)
  0.000153 seconds (64 CPU allocations: 1.875 KiB) (1 GPU allocation: 1024.000 KiB, 1.47% gc time)
=#

GC.gc(true)
if true
@ctime ∇softmax!(dx,dy,x,y;dims=1)
@ctime _∇softmax!(dx,dy,x,y;dims=1)
@ctime ∇softmax!(dx,dy,x,y;dims=2)
@ctime _∇softmax!(dx,dy,x,y;dims=2)
@ctime ∇softmax!(dx,dy,x,y;dims=:)
@ctime _∇softmax!(dx,dy,x,y;dims=:)
end

#=
∇softmax!(dx, dy, x, y; dims = 1)
  39.541 μs (25 allocations: 736 bytes)
  0.000077 seconds (34 CPU allocations: 1.062 KiB)
_∇softmax!(dx, dy, x, y; dims = 1)
  103.781 μs (50 allocations: 1.44 KiB)
  0.000090 seconds (59 CPU allocations: 1.781 KiB) (2 GPU allocations: 1.001 MiB, 2.14% gc time)
∇softmax!(dx, dy, x, y; dims = 2)
  121.838 μs (25 allocations: 736 bytes)
  0.000141 seconds (34 CPU allocations: 1.062 KiB)
_∇softmax!(dx, dy, x, y; dims = 2)
  52.789 μs (54 allocations: 1.52 KiB)
  0.000092 seconds (63 CPU allocations: 1.859 KiB) (2 GPU allocations: 1.004 MiB, 2.39% gc time)
∇softmax!(dx, dy, x, y; dims = (:))
  293.816 μs (25 allocations: 736 bytes)
  0.000323 seconds (32 CPU allocations: 1.031 KiB)
_∇softmax!(dx, dy, x, y; dims = (:))
  61.927 μs (43 allocations: 1.23 KiB)
  0.000096 seconds (50 CPU allocations: 1.547 KiB) (1 GPU allocation: 1024.000 KiB, 1.38% gc time)
=#

GC.gc(true)
if true
@ctime ∇logsoftmax!(dx,dy,x,y;dims=1)
@ctime _∇logsoftmax!(dx,dy,x,y;dims=1)
@ctime ∇logsoftmax!(dx,dy,x,y;dims=2)
@ctime _∇logsoftmax!(dx,dy,x,y;dims=2)
@ctime ∇logsoftmax!(dx,dy,x,y;dims=:)
@ctime _∇logsoftmax!(dx,dy,x,y;dims=:)
end

#=
∇logsoftmax!(dx, dy, x, y; dims = 1)
  39.390 μs (25 allocations: 736 bytes)
  0.000082 seconds (34 CPU allocations: 1.062 KiB)
_∇logsoftmax!(dx, dy, x, y; dims = 1)
  65.409 μs (66 allocations: 4.25 KiB)
  0.000156 seconds (75 CPU allocations: 4.594 KiB) (1 GPU allocation: 1024 bytes, 1.59% gc time)
∇logsoftmax!(dx, dy, x, y; dims = 2)
  92.615 μs (25 allocations: 736 bytes)
  0.000164 seconds (34 CPU allocations: 1.062 KiB)
_∇logsoftmax!(dx, dy, x, y; dims = 2)
  40.642 μs (70 allocations: 4.33 KiB)
  0.000091 seconds (79 CPU allocations: 4.672 KiB) (1 GPU allocation: 4.000 KiB, 2.18% gc time)
∇logsoftmax!(dx, dy, x, y; dims = (:))
  311.676 μs (25 allocations: 736 bytes)
  0.000396 seconds (32 CPU allocations: 1.031 KiB)
_∇logsoftmax!(dx, dy, x, y; dims = (:))
  85.144 μs (59 allocations: 3.36 KiB)
  0.000136 seconds (66 CPU allocations: 3.672 KiB)
=#

nothing
