NVCCFLAGS=-O3 --use_fast_math -Wno-deprecated-gpu-targets --compiler-options '-fPIC -fopenmp -O3 -Wall'
GPPFLAGS= -O3 -Wall -fPIC -fopenmp

all: libknet8.so

libknet8.so: cuda1.o cuda01.o cuda11.o cuda12.o cuda20.o cuda21.o im2col.o pooling.o
	nvcc --shared $(NVCCFLAGS) $^ -o $@

%.o: %.cu
	nvcc -c $(NVCCFLAGS) $< -o $@

%.o: %.cpp
	nvcc -c $(NVCCFLAGS) $< -o $@

# libknet8cpu.so: im2col.o pooling.o
# 	g++ $(GPPFLAGS) -shared $^ -o $@

# %.o: %.cpp
# 	g++ -c $(GPPFLAGS) $< -o $@

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
