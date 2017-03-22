NVCC := $(shell nvcc --version > /dev/null 2>&1; echo $$?)
FOMP := $(shell $(CXX) -fopenmp -c conv.cpp -o /dev/null > /dev/null 2>&1; echo $$?)
CFLAGS := -O3 -Wall -fPIC
OBJ := conv.o

ifeq ($(FOMP),0)
  CFLAGS := $(CFLAGS) -fopenmp
endif

ifeq ($(NVCC),0)
  CXX := nvcc
  CFLAGS := -O3 --use_fast_math -Wno-deprecated-gpu-targets --compiler-options "$(CFLAGS)"
  OBJ := cuda1.o cuda01.o cuda11.o cuda12.o cuda13.o cuda14.o cuda16.o cuda17.o cuda20.o cuda21.o $(OBJ)
endif

libknet8.so: $(OBJ)
	$(CXX) --shared $(CFLAGS) $^ -o $@

%.o: %.cu
	$(CXX) -c $(CFLAGS) $< -o $@

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

cuda1.cu: cuda1.jl unary.jl
	julia $< > $@

cuda01.cu: cuda01.jl broadcast.jl
	julia $< > $@

cuda11.cu: cuda11.jl broadcast.jl
	julia $< > $@

cuda12.cu: cuda12.jl broadcast.jl
	julia $< > $@

cuda13.cu: cuda13.jl broadcast.jl
	julia $< > $@

cuda14.cu: cuda14.jl broadcast.jl
		julia $< > $@

cuda16.cu: cuda16.jl broadcast.jl
		julia $< > $@

cuda17.cu: cuda17.jl broadcast.jl
		julia $< > $@

cuda20.cu: cuda20.jl reduction.jl
	julia $< > $@

cuda21.cu: cuda21.jl reduction.jl
	julia $< > $@

clean:
	-rm *.o *.cu libknet8.so

.PRECIOUS: %.cu
