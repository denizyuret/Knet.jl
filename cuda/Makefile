CFLAGS=-O3 -use_fast_math

libknet8.so: cuda1gen.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -o $@

%.o: %.cu
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

%.cu: %.jl
	julia $< > $@

cuda1gen.cu: cuda1gen.jl cuda1arg.jl

clean:
	-rm *.o *.cu libknet1.so
