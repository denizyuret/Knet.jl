# broadcasting vector y(w) to a matrix x(w,k)
# y is expected to be a vector to broadcast over x
# this kernel tries to take advantage of data reuse of broadcast operation
# y vector devided by BLOCK_SIZE number of elements between each thread block,
# each thread block iterates over second dimension of matrix,
#so from (1,1) to (BLOCK_SIZE,k) belongs to first thread block
# if number of thread block

#performance limitations
#if size of vector y smaller than BLOCK_SIZE_x,
#(BLOCK_SIZE_y*(BLOCK_SIZE_x-size(y))) number of thread from each block will be idle

#                         for high performance we need > 42 thread block (14 warp*3)
# thread_block_count = size(y)/BLOCK_SIZE_y > 42
# size(y) > BLOCK_SIZE_y*42
# with BLOCK_SIZE=32, broadcast_dim_size should be > 1344
# so if we have less than 1344 elements in vector y, performance will sour
# we can call lower limit 672, for better optimisation,(TODO-enis) I will make performance tests
# worst 448
# (sliding factor is introduced for this problem)

# this kernel can handle all vector and matrix dimensions

# explanation of kernel code is not added to prevent increase size of cuda14.cu
# (TODO-enis) provide a link to explanation of kernel index calculations for development

using Knet: broadcast_ops

function cuda14src(f, j=f, ex="$f(xi,yi)")
  sprint() do s
    print(s,"#define BLOCK_SIZE_x 32\n#define BLOCK_SIZE_y 32\n")
    for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
        print(s,

"""

__global__ void _$(F)_14($T *x, $T *y,$T *z, int firstdimsize, int x_N)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    #if (__CUDA_ARCH__ < 300 )
      __shared__ $T Ys[BLOCK_SIZE_y];
    #endif

    int index_x = BLOCK_SIZE_x*bx+tx;

    while((index_x)<firstdimsize)
    {
      #if (__CUDA_ARCH__ >= 300 )
        int laneId = threadIdx.x & 0x1f;
        int value;
        if (laneId == 0)    // all threads except lane 0, like ty==0
            {
              value = y[index_x];// first thread in each wrap loads one element
            }
        value = __shfl(value, 0);   // Get "value" from lane 0

      #else
        if( ty==0 )
        {
            Ys[tx]=y[index_x];
        }
        __syncthreads();
      #endif

      int Start = (ty * firstdimsize) + index_x;
      int Step = firstdimsize * BLOCK_SIZE_y;

        for (int k= Start; k<x_N; k+=Step)
        {
            $T xi = x[k];
            #if (__CUDA_ARCH__ >= 300 )
              $T yi = value;
            #else
              $T yi = Ys[tx];
            #endif
            z[k]=$ex;
        }
        index_x += BLOCK_SIZE_x*gridDim.x;
    }
}

extern "C" {
  void $(F)_14($T *x,$T *y,$T *z, int firstdimsize, int x_N) {
    int n_block = (firstdimsize+BLOCK_SIZE_x-1)/BLOCK_SIZE_x;
    dim3 dimGrid(n_block, 1);
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    //x_N size of the x
    //firstdimsize is size of y
    _$(F)_14<<<n_block,dimBlock>>>(x,y,z,firstdimsize,x_N);
  }
}
""")
    end
  end
end

for a in broadcast_ops
    if !isa(a,Tuple); a=(a,); end
    print(cuda14src(a...))
end
