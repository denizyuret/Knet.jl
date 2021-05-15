# cuda16 # Array,Array->Array (Multi dimensional broadcast up to some dimension with loop unrolling)
fp = open("cuda16.cu","w")
#using Knet: binary_ops

function cuda16src(f, j=f, ex="$f(xi,yi)")
  sprint() do s
    # it can handle arrays with 3 to 5 dimensions
    for dim_count=3:5
      for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
          print(s,"__global__ void _$(F)_16_$(dim_count)($T *x,$T *y, $T *z,")
          # place input variables
          for counter=0:dim_count-1
            print(s,"int stridex_$counter,")
          end
          for counter=0:dim_count-1
            print(s,"int stridey_$counter,")
          end
          for counter=0:dim_count-1
            print(s,"int stridez_$counter,")
          end
          # z is global index calculated from block and tread id
          print(s,

              """int N_z) {
                  int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
                  int index_x,index_y;
              """)

              print(s,
              """
                  int coords[$(dim_count)];

                  while (index_z < N_z) {
                      int temp_index = index_z;
              """)
              for counter=dim_count-1:-1:0
                print(s,"\n\tcoords[$counter] = temp_index / stridez_$counter;")
                print(s,"\n\ttemp_index = temp_index % stridez_$counter;")
              end
              print(s,
              """\n
                      index_x =0;
                      index_y = 0;
              """)
              for counter=0:dim_count-1
                print(s,"\n\tindex_x+= stridex_$counter*coords[$counter];")
                print(s,"\n\tindex_y+= stridey_$counter*coords[$counter];")
              end
              print(s,
              """\n
                      $T xi = x[index_x];
                      $T yi = y[index_y];
                      z[index_z]=$ex;
                      //z[index_z]=index_z;
                      index_z+=blockDim.x * gridDim.x;

                  }
              }

              extern "C" {\n""")
              for stream in (false,true)
              if stream
                  print(s, "$DLLEXPORT void $(F)_16_$(dim_count)_stream($T *x,$T *y,$T *z,")
              else
                  print(s, "$DLLEXPORT void $(F)_16_$(dim_count)($T *x,$T *y,$T *z,")
              end
              for counter=0:dim_count-1
                print(s,"int stridex_$counter,")
              end
              for counter=0:dim_count-1
                print(s,"int stridey_$counter,")
              end
              for counter=0:dim_count-1
                print(s,"int stridez_$counter,")
              end
              if stream
              print(s,
                """int Nz, cudaStream_t STR) {
                  _$(F)_16_$(dim_count)<<<256,256,0,STR>>>(x,y,z,""")
              else
              print(s,
                """int Nz) {
                  _$(F)_16_$(dim_count)<<<256,256>>>(x,y,z,""")
              end
              for counter=0:dim_count-1
                print(s,"stridex_$counter,")
              end
              for counter=0:dim_count-1
                print(s,"stridey_$counter,")
              end
              for counter=0:dim_count-1
                print(s,"stridez_$counter,")
              end
              print(s,
                    """Nz);
                }
              """)
        end
        print(s,"}\n")
      end
    end
  end
end

for a in binary_ops
    if !isa(a,Tuple); a=(a,); end
    print(fp,cuda16src(a...))
end

close(fp)
