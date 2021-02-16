NVCC := "nvcc"
NVCCFLAGS :=

CFLAGS := -O3 -Wall -fPIC
OBJ := conv.o

NO_NVCC := $(shell $(NVCC) --version > /dev/null 2>&1; echo $$?)
NO_FOMP := $(shell $(CXX) -fopenmp -c conv.cpp -o /dev/null > /dev/null 2>&1; echo $$?)

ifeq ($(NO_FOMP),0)
  CFLAGS += -fopenmp
endif

ifeq ($(NO_NVCC),0)
  CXX := $(NVCC)
  CFLAGS := -O3 --use_fast_math -Wno-deprecated-gpu-targets $(NVCCFLAGS) --compiler-options "$(CFLAGS)"
  OBJ := cuda1.o cuda01.o cuda11.o cuda12.o cuda13.o cuda16.o cuda17.o cuda20.o cuda21.o cuda22.o $(OBJ)
endif

libknet8.so: $(OBJ)
	$(CXX) --shared $(CFLAGS) $^ -o $@

%.o: %.cu
	$(CXX) -c $(CFLAGS) $< -o $@

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

cuda1.cu: cuda1.jl unary.jl
	julia $< > $@

cuda01.cu: cuda01.jl binary.jl
	julia $< > $@

cuda11.cu: cuda11.jl binary.jl
	julia $< > $@

cuda12.cu: cuda12.jl binary.jl
	julia $< > $@

cuda13.cu: cuda13.jl binary.jl
	julia $< > $@


cuda16.cu: cuda16.jl binary.jl
		julia $< > $@

cuda17.cu: cuda17.jl binary.jl
		julia $< > $@

cuda20.cu: cuda20.jl reduction.jl
	julia $< > $@

cuda21.cu: cuda21.jl reduction.jl
	julia $< > $@

cuda22.cu: cuda22.jl reduction.jl
	julia $< > $@

clean:
	-rm *.o *.cu *.obj libknet8.*

.PRECIOUS: %.cu
