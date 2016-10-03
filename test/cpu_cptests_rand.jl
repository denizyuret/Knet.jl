using Base.Test
using CUDArt
using CUDNN
@show CUDNN_VERSION

import CUDNN.cudnnConvolutionForward
import CUDNN.cudnnConvolutionBackwardFilter
import CUDNN.cudnnConvolutionBackwardData
import CUDNN.cudnnGetConvolutionNdForwardOutputDim
import CUDNN.cudnnGetPoolingNdForwardOutputDim
import CUDNN.cudnnPoolingForward
import CUDNN.cudnnPoolingBackward
include(Knet.dir("src","util","conv_pool_cpu.jl"))

padding=0
stride=1
x = rand(Float32,64,64,3,128); tx = CudaArray(x)
w = rand(Float32,8,8,3,10); tw = CudaArray(w)

dw = zeros(Float32, size(w)); tdw = CudaArray(dw);
dx = zeros(Float32, size(x)); tdx = CudaArray(dx);

dw2 = zeros(Float32, size(w))
dx2 = zeros(Float32, size(x));

y = zeros(Float32, cudnnGetConvolutionNdForwardOutputDim(tx,tw; padding=padding,stride=stride)); ty = CudaArray(y)
y2 = zeros(Float32, size(y))
dy = rand(Float32,size(y)); tdy = CudaArray(dy)


cudnnConvolutionForward(tx,tw,ty; padding=padding, stride=stride); y = to_host(ty)
@time cudnnConvolutionForward(x,w,y2; padding=padding, stride=stride);
@test_approx_eq y y2

cudnnConvolutionBackwardFilter(tx,tdy,tdw); dw = to_host(tdw)
@time cudnnConvolutionBackwardFilter(x,dy,dw2)
@test_approx_eq dw dw2

cudnnConvolutionBackwardData(tw, tdy, tdx); dx = to_host(tdx)
@time cudnnConvolutionBackwardData(w, dy, dx2);
@test_approx_eq dx dx2


using CUDNN: PD, CUDNN_POOLING_MAX, cudnnGetPoolingNdForwardOutputDim
x = rand(Float32,64,64,3,128); tx = CudaArray(x);
psize, padding, stride = 5, 0, 5
pd1 = PD(2, psize, padding, stride, CUDNN_POOLING_MAX)
@assert cudnnGetPoolingNdForwardOutputDim(pd1, tx) == cudnnGetPoolingNdForwardOutputDim(x, window=psize, padding=padding, stride=stride, mode=0)
@show ydims = cudnnGetPoolingNdForwardOutputDim(x, window=psize, padding=padding, stride=stride, mode=0)
y = zeros(Float32, ydims); ty = CudaArray(y);
y2 = zeros(y)
cudnnPoolingForward(tx, ty; window=psize, padding=padding, stride=stride, mode=0); y = to_host(ty);
@time cudnnPoolingForward(x, y2; window=psize, padding=padding, stride=stride, mode=0);
@test_approx_eq y y2

dx = zeros(Float32, size(x)); tdx = CudaArray(dx);
dx2 = zeros(Float32, size(x));
dy = rand(Float32, size(y)); tdy = CudaArray(dy);
cudnnPoolingBackward(ty, tdy, tx, tdx; window=psize, padding=padding, stride=stride, mode=0); dx = to_host(tdx);
@time cudnnPoolingBackward(y, dy, x, dx2; window=psize, padding=padding, stride=stride, mode=0);
@test_approx_eq dx dx2

:ok
