# Multi dimensional array broadcast
# this kernel can handle all arrays with different dimensions and broadcasting multiple dimensions
# x and y expected to be an array
#in stride_x and stride_y, values corresponding to broadcasting dim should be zero
#in the one that is being broadcasted
# N_z is the size of the z array
# dimlen_z is the number of the dimensions in z

#performance limitations
#stride values and coordinates kept as arrays, causing them to replaced in heap,shared memory
# (slow: http://stackoverflow.com/a/13485322)
# TODO-enis understand heap memory and dynamic allocation for possible bug cause with arrays of >100 dims
# http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heap-memory-allocation

fp = open("cuda17.cu","w")
using Knet: broadcast_ops

function cuda17src(f, j=f, ex="$f(xi,yi)")
  sprint() do s

    for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
        print(s,

"""
__global__ void _$(F)_17($T *x,$T *y, $T *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        $T xi = x[index_x];
        $T yi = y[index_y];
        z[index_z]=$ex;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void $(F)_17($T *x,$T *y,$T *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _$(F)_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
""")
    end
  end
end

for a in broadcast_ops
    if !isa(a,Tuple); a=(a,); end
    print(fp,cuda17src(a...))
end

close(fp)
