cuda44permutedims3D = [
#("permutedims3D_1_2_3","i","j","k"),#noop
("permutedims3D_1_3_2","i","k","j"),
("permutedims3D_2_1_3","j","i","k"),
("permutedims3D_2_3_1","k","i","j"),
("permutedims3D_3_1_2","j","k","i"),
("permutedims3D_3_2_1","k","j","i"),
]

function cuda44permutedims3Dsrc(f, i1, i2, i3; BLK=256, THR=256)
    sprint() do s
        for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
            print(s,
"""
__global__ void _$(F)_44($T* x, int dimx1, int dimx2, int dimx3, $T* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimy1*dimy2*dimy3; v += blockDim.x * gridDim.x) {

		int i = v % dimy1;
		int j = ((v - i) / dimy1) % dimy2;
		int k = ((v - j * dimy1 - i) / (dimy1 * dimy2)) % dimy3;

		int srcIndex = $i1 + dimx1*$i2 + dimx1*dimx2*$i3;
		y[v] = x[srcIndex];
	}
}
extern "C" {
  void $(F)_44($T* x, int dimx1, int dimx2, int dimx3, $T* y, int dimy1, int dimy2, int dimy3) {
    _$(F)_44<<<$BLK,$THR>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2,dimy3);
  }    
}
""")
        end
    end
end

for a in cuda44permutedims3D
    isa(a,Tuple) || (a=(a,))
    print(cuda44permutedims3Dsrc(a...))
end