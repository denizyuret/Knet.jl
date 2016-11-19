using Knet: cudaProperties

function cudapsrc(f, j, ex)
  sprint() do s
    print(s,
"""
extern "C"
{
  int $(f)(int id)
  {
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties,id) == 0)
      return properties.$ex;
    else
      return -1;
  }
}
"""
    )
  end
end

for a in cudaProperties
  isa(a,Tuple) || (a=(a,))
  print(cudapsrc(a...))
end