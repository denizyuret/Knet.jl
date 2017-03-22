# broadcasting vector y(1,1,1,w,1...) to an N-dim array x(x,y,z,w,t...)
# y is expected to be a vector to broadcast over x
# this kernel tries to take advantage of data reuse of broadcast operation
# y vector devided by BLOCK_SIZE number of elements between each thread block,
# lets say dimensions are grouped by being lower or higher than broadcast dimension
# for y(1,1,z,1,1...) and x(x,y,z,w,t...), x,y are lower and w,t is higher dims
# each thread block operates over all elements in lower dims
#corresponding to BLOCK_SIZE number of elements in broadcast dimension
#so from (1,1,1,1,1) to (x,y,BLOCK_SIZE,1,1) belongs to first thread block

#performance limitations
#if broadcast_dim_stride smaller than BLOCK_SIZE_x,
#it will cause some of the threads to stay idle,

#                         for high performance we need > 42 thread block (14 warp*3)
# thread_block_count = broadcast_dim_size/BLOCK_SIZE_y > 42
# broadcast_dim_size > BLOCK_SIZE_y*42
# with BLOCK_SIZE=32, broadcast_dim_size should be > 1344
# so if we have less than 1344 elements in broadcast dim, performance will sour
# we can call lower limit 672, for better optimisation,(TODO-enis) I will make performance tests
#  worst 448

# this kernel can handle vector size of 65535*32 = 2.097.120 elements
# handling everything might have cause extra overheat, TODO-enis add support for limitless

# explanation of kernel code is not added to prevent increase size of cuda13.cu
# (TODO-enis) provide a link to explanation of kernel index calculations for development


using Knet: broadcast_ops

function cuda13src(f, j=f, ex="$f(xi,yi)")
  sprint() do s
    print(s,"#define BLOCK_SIZE 32")
    for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
        print(s,

"""

__global__ void _$(F)_13($T *x,$T *y,$T *z, int brdcastdimstride, int brdcastnextstride,int multidimsize) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ $T Bs[BLOCK_SIZE];
    if( ty==0 )
    {
        int vector_index = BLOCK_SIZE*bx+tx;
        Bs[tx]=y[vector_index];
    }
    __syncthreads();
    int Start = (((32*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE;
    if (tx<brdcastdimstride)
    {
        for (int k=0; k< multidimsize; k++)
        {
            for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
            {
                $T xi = x[i];
                $T yi = Bs[ty];
                z[i]=$ex;
            }
            Start +=brdcastnextstride;
        }
    }
}
extern "C" {
  void $(F)_13($T *x,$T *y,$T *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int B_N) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int n_block = (B_N+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 dimGrid(n_block);
    _$(F)_13<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize);
  }
}
""")
    end
  end
end

for a in broadcast_ops
    if !isa(a,Tuple); a=(a,); end
    print(cuda13src(a...))
end
