NVCC=nvcc
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70)

NVCC_FLAGS += -dc -Xcompiler -fopenmp -lineinfon $(GENCODE_FLAGS) -std=c++14 

NVCC_LDFLAGS = -lcuda -lcudart -Xcompiler -fopenmp

Filter: Makefile vectoradd.cu
	$(NVCC) $(NVCC_FLAGS) vectoradd.cu -c -o vectoradd.o
	$(NVCC) $(GENCODE_FLAGS) vectoradd.o -o vectoradd $(NVCC_LDFLAGS)

.PHONY.: clean
clean:
	rm -f  vectoradd.o  vectoradd
