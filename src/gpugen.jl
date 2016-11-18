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
    cudaGetDeviceProperties(&properties,id);
    return properties.$ex;
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