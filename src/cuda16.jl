
using Knet: broadcast_ops

function cuda16src(f, j=f, ex="$f(xi,yi)")
  sprint() do s
    # it can handle arrays with 3 to 10 dimensions
    for dim_count=3:10
      for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
          print(s,"__global__ void _$(F)_16_$(dim_count)($T *x,$T *y, $T *z,")
          for counter=1:dim_count
            print(s,"int stridex_$counter,")
            print(s,"int stridey_$counter,")
            print(s,"int stridez_$counter,")
          end
          print(s,

              """int N_z) {
                  int index_z = threadIdx.x + blockIdx.x * blockDim.x;
                  int index_x,index_y;
                  int coords[$(dim_count)];

                  while (index_z < N_z) {
                      int temp_index = index_z;
              """)
              for counter=0:dim_count-1
                print(s,"\n\tcoords[$counter] = temp_index / stride_z[$counter];")
                print(s,"\n\ttemp_index = temp_index % stride_z[$counter];")
              end
              print(s,
              """\n
                      index_x =0;
                      index_y = 0;
              """)
              for counter=0:dim_count-1
                print(s,"\n\tindex_x+= stride_x[$counter]*coords[$counter];")
                print(s,"\n\tindex_y+= stride_y[$counter]*coords[$counter];")
              end
              print(s,
              """\n
                      $T xi = x[index_x];
                      $T yi = y[index_y];
                      z[index_z]=$ex;
                      index_z+=blockDim.x * gridDim.x;

                  }
              }

              extern "C" {
                void $(F)_16_$(dim_count)($T *x,$T *y,$T *z,""")
              for counter=1:dim_count
                print(s,"int stridex_$counter,")
                print(s,"int stridey_$counter,")
                print(s,"int stridez_$counter,")
              end

              print(s,
                """ int Nz) {

                  _$(F)_16_$(dim_count)<<<256,256>>>(x,y,z,""")
              for counter=1:dim_count
                print(s,"stridex_$counter,")
                print(s,"stridey_$counter,")
                print(s,"stridez_$counter,")
              end
              print(s,
                    """size1,Nz);
                }
              }
              """)
      end
    end
  end)
end

for a in broadcast_ops
    if !isa(a,Tuple); a=(a,); end
    print(cuda16src(a...))
end
