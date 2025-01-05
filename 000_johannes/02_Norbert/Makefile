NVCC=nvcc
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM72    := -gencode arch=compute_72,code=sm_72
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM72)

NVCC_FLAGS += -dc -Xcompiler -fopenmp -lineinfon $(GENCODE_FLAGS) -std=c++14

NVCC_LDFLAGS = -lcuda -lcudart -Xcompiler -fopenmp

SRCDIR = ./src
OUTDIR = ./bin

all: $(OUTDIR)/Aufgabe2_shared_cudaMallocHost $(OUTDIR)/Aufgabe2_shared_malloc $(OUTDIR)/Aufgabe2_simple_cudaMallocHost $(OUTDIR)/Aufgabe2_simple_malloc
$(OUTDIR)/Aufgabe2_shared_cudaMallocHost: $(SRCDIR)/Aufgabe2_shared_cudaMallocHost.cu
	mkdir -p $(OUTDIR)
	$(NVCC) $(NVCC_FLAGS) $(SRCDIR)/Aufgabe2_shared_cudaMallocHost.cu -c -o $(OUTDIR)/Aufgabe2_shared_cudaMallocHost.o
	$(NVCC) $(GENCODE_FLAGS) $(OUTDIR)/Aufgabe2_shared_cudaMallocHost.o -o $(OUTDIR)/Aufgabe2_shared_cudaMallocHost $(NVCC_LDFLAGS)


$(OUTDIR)/Aufgabe2_shared_malloc: $(SRCDIR)/Aufgabe2_shared_malloc.cu
	mkdir -p $(OUTDIR)
	$(NVCC) $(NVCC_FLAGS) $(SRCDIR)/Aufgabe2_shared_malloc.cu -c -o $(OUTDIR)/Aufgabe2_shared_malloc.o
	$(NVCC) $(GENCODE_FLAGS) $(OUTDIR)/Aufgabe2_shared_malloc.o -o $(OUTDIR)/Aufgabe2_shared_malloc $(NVCC_LDFLAGS)

$(OUTDIR)/Aufgabe2_simple_cudaMallocHost: $(SRCDIR)/Aufgabe2_simple_cudaMallocHost.cu
	mkdir -p $(OUTDIR)
	$(NVCC) $(NVCC_FLAGS) $(SRCDIR)/Aufgabe2_simple_cudaMallocHost.cu -c -o $(OUTDIR)/Aufgabe2_simple_cudaMallocHost.o
	$(NVCC) $(GENCODE_FLAGS) $(OUTDIR)/Aufgabe2_simple_cudaMallocHost.o -o $(OUTDIR)/Aufgabe2_simple_cudaMallocHost $(NVCC_LDFLAGS)

$(OUTDIR)/Aufgabe2_simple_malloc: $(SRCDIR)/Aufgabe2_simple_malloc.cu
	mkdir -p $(OUTDIR)
	$(NVCC) $(NVCC_FLAGS) $(SRCDIR)/Aufgabe2_simple_malloc.cu -c -o $(OUTDIR)/Aufgabe2_simple_malloc.o
	$(NVCC) $(GENCODE_FLAGS) $(OUTDIR)/Aufgabe2_simple_malloc.o -o $(OUTDIR)/Aufgabe2_simple_malloc $(NVCC_LDFLAGS)

.PHONY.: clean
clean:
	rm -f  $(OUTDIR)/*