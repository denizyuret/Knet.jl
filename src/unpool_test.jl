using Knet;

x = KnetArray(reshape(Float32[1.0:16.0...], (4,4,1,1)))
y = pool(x)

#=
x (2x2)
6   14
8   16
unpool (2x2) window
6   6   14  14
6   6   14  14
8   8   16  16
8   8   16  16
unpool (3x3 window)
6   6   6   14  14  14
6   6   6   14  14  14
6   6   6   14  14  14
6   6   6   14  14  14
6   6   6   14  14  14
6   6   6   14  14  14
=#

display(myunpool(y))
display(myunpool(y;window=3))