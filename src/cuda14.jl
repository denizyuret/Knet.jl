# broadcasting vector y(w) to a N-dim array x(w,k,m,l...)
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

# shufle is slow due to index Access pattern, so removed

using Knet: broadcast_ops

function cuda14src(f, j=f, ex="$f(xi,yi)")
  sprint() do s
    print(s,"#define BLOCK_SIZE_x 32\n#define BLOCK_SIZE_y 32\n")
    for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
        print(s,

"""

__global__ void _$(F)_14_x_y($T *x, $T *y,$T *z, int firstdimsize, int x_N)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ $T Ys[BLOCK_SIZE_y];
    int index_x = BLOCK_SIZE_x*bx+tx;

    while((index_x)<firstdimsize)
    {
      if( ty==0 )
      {
          Ys[tx]=y[index_x];
      }
      __syncthreads();
      int Start = (ty * firstdimsize) + index_x;
      int Step = firstdimsize * BLOCK_SIZE_y;

      for (int k= Start; k<x_N; k+=Step)
      {
          $T xi = x[k];
          $T yi = Ys[tx];
          z[k]=$ex;
      }
      index_x += BLOCK_SIZE_x*gridDim.x;
    }
}

extern "C" {
  void $(F)_14_x_y($T *x,$T *y,$T *z, int firstdimsize, int x_N) {
    int n_block = (firstdimsize+BLOCK_SIZE_x-1)/BLOCK_SIZE_x;
    dim3 dimGrid(n_block, 1);
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    //x_N size of the x
    //firstdimsize is size of y
    _$(F)_14_x_y<<<n_block,dimBlock>>>(x,y,z,firstdimsize,x_N);
  }
}
""")
    end


    for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
        print(s,

"""

__global__ void _$(F)_14_y_x($T *x, $T *y,$T *z, int firstdimsize, int x_N)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ $T Ys[BLOCK_SIZE_y];
    int index_x = BLOCK_SIZE_x*bx+tx;

    while((index_x)<firstdimsize)
    {
      if( ty==0 )
      {
          Ys[tx]=y[index_x];
      }
      __syncthreads();
      int Start = (ty * firstdimsize) + index_x;
      int Step = firstdimsize * BLOCK_SIZE_y;

      for (int k= Start; k<x_N; k+=Step)
      {
          $T yi = x[k];
          $T xi = Ys[tx];
          z[k]= $ex;
      }
      index_x += BLOCK_SIZE_x*gridDim.x;
    }
}

extern "C" {
  void $(F)_14_y_x($T *x,$T *y,$T *z, int firstdimsize, int x_N) {
    int n_block = (firstdimsize+BLOCK_SIZE_x-1)/BLOCK_SIZE_x;
    dim3 dimGrid(n_block, 1);
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    //x_N size of the x
    //firstdimsize is size of y
    _$(F)_14_y_x<<<n_block,dimBlock>>>(x,y,z,firstdimsize,x_N);
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
