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

# Seperate Kernels
$(OBJDIR)/01_seperate_kernels: $(OBJDIR)/01_seperate_kernels.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

# Atomic Kernel
$(OBJDIR)/02_atomic_kernel: $(OBJDIR)/02_atomic_kernel.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

# Atomic Kernel with cascade
$(OBJDIR)/03_atomic_kernel_cascade: $(OBJDIR)/03_atomic_kernel_cascade.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

# Atomic Kernel without modulo operations and optimized control flow
$(OBJDIR)/04_atomic_kernel_opt: $(OBJDIR)/04_atomic_kernel_opt.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

# Atomic Kernel with sequential addressing
$(OBJDIR)/05_atomic_kernel_seq_addressing: $(OBJDIR)/05_atomic_kernel_seq_addressing.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

# Atomic Kernel with loop unrolling
$(OBJDIR)/06_atomic_kernel_loop_unroll: $(OBJDIR)/06_atomic_kernel_loop_unroll.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

# Atomic Kernel with shuffeling
$(OBJDIR)/07_atomic_kernel_shuffle: $(OBJDIR)/07_atomic_kernel_shuffle.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

INPUT_SIZE=$(shell echo  2 ^ 21 | bc)
NVPROF=nsys profile --stats=true 
all: $(OBJDIR)/01_seperate_kernels \
	 $(OBJDIR)/02_atomic_kernel \
	 $(OBJDIR)/03_atomic_kernel_cascade \
	 $(OBJDIR)/04_atomic_kernel_opt \
	 $(OBJDIR)/05_atomic_kernel_seq_addressing \
	 $(OBJDIR)/06_atomic_kernel_loop_unroll \
	 $(OBJDIR)/07_atomic_kernel_shuffle
	@echo "Expected output: $(INPUT_SIZE)"
	#$(NVPROF) $(OBJDIR)/01_seperate_kernels $(INPUT_SIZE)
	#$(NVPROF) $(OBJDIR)/02_atomic_kernel $(INPUT_SIZE)
	#$(NVPROF) $(OBJDIR)/03_atomic_kernel_cascade $(INPUT_SIZE)
	#$(NVPROF) $(OBJDIR)/04_atomic_kernel_opt $(INPUT_SIZE)
	#$(NVPROF) $(OBJDIR)/05_atomic_kernel_seq_addressing $(INPUT_SIZE)
	$(NVPROF) $(OBJDIR)/06_atomic_kernel_loop_unroll $(INPUT_SIZE)
	$(NVPROF) $(OBJDIR)/07_atomic_kernel_shuffle $(INPUT_SIZE)
	rm -rf report*.nsys-rep report*.sqlite

.PHONY.: clean
clean:
	rm -rf $(OBJDIR)
	rm -rf report*.nsys-rep report*.sqlite