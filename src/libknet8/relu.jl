fp = open("relu.cu","w")

function relusrc(; BLK=256, THR=256)
  sprint() do s
    for (T,F,Fback) in [("float","relu_32_1","reluback_32_1"),("double","relu_64_1","reluback_64_1")]
        print(s,
"""
__global__ void _$(F)(int n, $T max_value, $T negative_slope, $T threshold, $T *x_, $T *y_) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    $T x = x_[i];
    y_[i] = (x >= max_value ? max_value : x >= threshold ? x : negative_slope == 0 ? 0 : negative_slope * (x-threshold));
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _$(Fback)(int n, $T max_value, $T negative_slope, $T threshold, $T *x_, $T *y_, $T *dy_, $T *dx_) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    $T x=x_[i];
    dx_[i] = dy_[i] * (x >= max_value ? 0 : x >= threshold ? 1 : negative_slope);
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  $DLLEXPORT void $F(int n, $T max_value, $T negative_slope, $T threshold, $T *x, $T *y) {
    if (n>0) _$F<<<$BLK,$THR>>>(n,max_value,negative_slope,threshold,x,y);
  }
  $DLLEXPORT void $(F)_stream(int n, $T max_value, $T negative_slope, $T threshold, $T *x, $T *y, cudaStream_t STR) {
    if (n>0) _$F<<<$BLK,$THR,0,STR>>>(n,max_value,negative_slope,threshold,x,y);
  }
  $DLLEXPORT void $(Fback)(int n, $T max_value, $T negative_slope, $T threshold, $T *x, $T *y, $T *dy, $T *dx) {
    _$(Fback)<<<$BLK,$THR>>>(n,max_value,negative_slope,threshold,x,y,dy,dx);
  }    
  $DLLEXPORT void $(Fback)_stream(int n, $T max_value, $T negative_slope, $T threshold, $T *x, $T *y, $T *dy, $T *dx, cudaStream_t STR) {
    _$(Fback)<<<$BLK,$THR,0,STR>>>(n,max_value,negative_slope,threshold,x,y,dy,dx);
  }    
}
""")
    end
  end
end

print(fp, relusrc())
close(fp)
