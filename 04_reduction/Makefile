NVCC=nvcc
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM72    := -gencode arch=compute_72,code=sm_72
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80
GENCODE_FLAGS	:= $(GENCODE_SM80)

NVCC_FLAGS += -dc -Xcompiler -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++14
NVCC_LDFLAGS = -ccbin=g++  -lcuda -lcudart -ljpeg

OBJDIR=./build

$(OBJDIR)/%.o: %.cu util.h
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJDIR)/%: $(OBJDIR)/%.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

KERNELS = $(OBJDIR)/01_seperate_kernels \
	 $(OBJDIR)/02_atomic_kernel \
	 $(OBJDIR)/03_atomic_kernel_cascade \
	 $(OBJDIR)/04_atomic_kernel_opt \
	 $(OBJDIR)/05_atomic_kernel_seq_address \
	 $(OBJDIR)/06_atomic_kernel_loop_unroll \
	 $(OBJDIR)/07_atomic_kernel_shuffle

all: $(KERNELS)

INPUT_SIZE=$(shell echo  2 ^ 21 | bc)
#NVPROF=nsys profile --stats=true --output="./build/report%n"
run: $(KERNELS)
	@echo "Expected output: $(INPUT_SIZE)"
	for kernel in $(KERNELS); do \
		echo "Running $$kernel"; \
		$(NVPROF) $$kernel $(INPUT_SIZE); \
	done

.PHONY.: clean
clean:
	rm -rf $(OBJDIR)
	rm -rf report*.nsys-rep report*.sqlite