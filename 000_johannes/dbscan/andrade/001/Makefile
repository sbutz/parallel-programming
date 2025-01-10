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
GENCODE_FLAGS	:= $(GENCODE_SM72)

NVCC_FLAGS += -dc -Xcompiler -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++14
NVCC_LDFLAGS = -ccbin=g++  -lcuda -lcudart -ljpeg

SRCDIR=./src
OBJDIR=./bin
NVIDIADIR=./nvidia

all: $(OBJDIR)/andrade

$(OBJDIR)/andrade: \
	$(OBJDIR)/andrade.o $(OBJDIR)/count_neighbors.o $(OBJDIR)/accumulate.o \
	$(OBJDIR)/build_incidence_lists.o $(OBJDIR)/build_graph.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/andrade.o: $(SRCDIR)/andrade.cc $(SRCDIR)/read_input.h $(SRCDIR)/build_graph.h

$(OBJDIR)/build_graph.o: $(SRCDIR)/build_graph.cc $(SRCDIR)/count_neighbors.h \
	$(SRCDIR)/accumulate.h $(SRCDIR)/build_incidence_lists.h $(SRCDIR)/cuda_helpers.h \
	$(SRCDIR)/build_graph.h

$(OBJDIR)/count_neighbors.o: $(SRCDIR)/count_neighbors.cu $(SRCDIR)/count_neighbors.h $(SRCDIR)/cuda_helpers.h

$(OBJDIR)/accumulate.o: $(SRCDIR)/accumulate.cu $(SRCDIR)/accumulate.h $(SRCDIR)/cuda_helpers.h

$(OBJDIR)/build_incidence_lists.o: $(SRCDIR)/build_incidence_lists.cu $(SRCDIR)/build_incidence_lists.h $(SRCDIR)/cuda_helpers.h \
	$(SRCDIR)/device_vector.h

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

.PHONY.: clean
clean:
	rm -rf $(OBJDIR)
