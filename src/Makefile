CFLAGS=-O3 -use_fast_math -Wno-deprecated-gpu-targets

libknet8.so: cuda1.o cuda01.o cuda11.o cuda12.o cuda20.o cuda21.o gpuh.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -o $@

%.o: %.cu
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

cuda1.cu: cuda1.jl unary.jl
	julia $< > $@

cuda01.cu: cuda01.jl broadcast.jl
	julia $< > $@

cuda11.cu: cuda11.jl broadcast.jl
	julia $< > $@

cuda12.cu: cuda12.jl broadcast.jl
	julia $< > $@

cuda20.cu: cuda20.jl reduction.jl
	julia $< > $@

cuda21.cu: cuda21.jl reduction.jl
	julia $< > $@

clean:
	-rm *.o *.cu libknet8.so

.PRECIOUS: %.cu
