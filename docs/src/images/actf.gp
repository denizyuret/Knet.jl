set size square
set nokey
set terminal png size 750,500
set output 'actf.png'
set multiplot layout 2,3 columnsfirst
set yrange [-1.4:1.4]
set xrange [-10:10]
set grid

set title '0/1 step'
plot 0 * (x<0) + 1 * (x >= 0)

set title 'sigmoid: 1/(1+e^{-x})'
plot 1/(1+exp(-x))

set title '-1/+1 step'
plot -1 * (x<0) + 1 * (x >= 0)

set title 'tanh: (e^x-e^{-x})/(e^x+e^{-x})'
plot tanh(x)

set xrange [-1.4:1.4]

set title 'relu: max(0,x)'
plot 0 * (x<0) + x * (x >= 0)
set title 'maxout'
plot (x<-0.5) * (-3*x-2) + (x>=-0.5) * (x<0.5) * (0.25*x-0.375) + (x >= 0.5) * (1.5*x-1)

