CFLAGS=-O3 -use_fast_math

libknet8.so: cuda1.o cuda01.o cuda10.o cuda11.o cuda12.o cuda20.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -o $@

%.o: %.cu
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

%.cu: %gen.jl %.jl
	julia $< > $@

clean:
	-rm *.o *.cu libknet8.so

.PRECIOUS: *.cu
