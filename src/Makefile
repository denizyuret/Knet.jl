ifneq ($(shell which nvcc),)
  CC=nvcc
  CFLAGS=-O3 --use_fast_math -Wno-deprecated-gpu-targets --compiler-options '-fPIC -fopenmp -O3 -Wall'
  OBJ=cuda1.o cuda01.o cuda11.o cuda12.o cuda20.o cuda21.o conv.o
else
  CC=g++
  CFLAGS=-O3 -Wall -fPIC -fopenmp
  OBJ=conv.o
endif

libknet8.so: $(OBJ)
	$(CC) --shared $(CFLAGS) $^ -o $@

%.o: %.cu
	$(CC) -c $(CFLAGS) $< -o $@

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

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
