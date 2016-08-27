CFLAGS=-O3 -I. --use_fast_math

libknet1.so: cuda1gen.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -o $@

%.o: %.cu
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

%.cu: %.jl
	julia $< > $@

cuda1gen.cu: cuda1gen.jl cuda1arg.jl
	julia $< > $@

clean:
	-rm *.o libknet1.so
