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
NVCC_LDFLAGS = -ccbin=g++  -lcuda -lcudart

SRCDIR=./src
OBJDIR=./bin
NVIDIADIR=./nvidia

all: $(OBJDIR)/deviceQuery \
$(OBJDIR)/a_andrade \
$(OBJDIR)/a_andrade_01 \
$(OBJDIR)/a_andrade_02 \
$(OBJDIR)/a_andrade_04 \
$(OBJDIR)/a_andrade_05 \
$(OBJDIR)/a_andrade_texture \
$(OBJDIR)/b_boehm \
$(OBJDIR)/b_boehm_2 \
$(OBJDIR)/b_boehm_4 \
$(OBJDIR)/b_boehm_8 \
$(OBJDIR)/b_boehm_16 \
$(OBJDIR)/b_boehm_32 \
$(OBJDIR)/b_boehm_64 \
$(OBJDIR)/b_boehm_128 \
$(OBJDIR)/b_boehm_256 \
$(OBJDIR)/b_boehm_512 \
$(OBJDIR)/b_boehm_1024 \
$(OBJDIR)/c_boehm

$(OBJDIR)/deviceQuery: $(OBJDIR)/deviceQuery.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/a_andrade: $(OBJDIR)/a_andrade.o $(OBJDIR)/warmup.o $(OBJDIR)/a_bfs.o $(OBJDIR)/a_build_graph.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/a_andrade_01: $(OBJDIR)/a_andrade.o $(OBJDIR)/warmup.o $(OBJDIR)/a_bfs_01_naive.o $(OBJDIR)/a_build_graph.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/a_andrade_02: $(OBJDIR)/a_andrade.o $(OBJDIR)/warmup.o $(OBJDIR)/a_bfs_02_succ.o $(OBJDIR)/a_build_graph.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/a_andrade_04: $(OBJDIR)/a_andrade.o $(OBJDIR)/warmup.o $(OBJDIR)/a_bfs_04_succmultwarp.o $(OBJDIR)/a_build_graph.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/a_andrade_05: $(OBJDIR)/a_andrade.o $(OBJDIR)/warmup.o $(OBJDIR)/a_bfs_05_frontiershared.o $(OBJDIR)/a_build_graph.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/a_andrade_texture: $(OBJDIR)/a_andrade.o $(OBJDIR)/warmup.o $(OBJDIR)/a_bfs_texture.o $(OBJDIR)/a_build_graph.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/b_boehm: $(OBJDIR)/b_boehm.o $(OBJDIR)/warmup.o $(OBJDIR)/b_cluster_expansion.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/b_boehm_2: $(OBJDIR)/b_boehm.o $(OBJDIR)/warmup.o $(OBJDIR)/b_cluster_expansion_2.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/b_boehm_4: $(OBJDIR)/b_boehm.o $(OBJDIR)/warmup.o $(OBJDIR)/b_cluster_expansion_4.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/b_boehm_8: $(OBJDIR)/b_boehm.o $(OBJDIR)/warmup.o $(OBJDIR)/b_cluster_expansion_8.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/b_boehm_16: $(OBJDIR)/b_boehm.o $(OBJDIR)/warmup.o $(OBJDIR)/b_cluster_expansion_16.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/b_boehm_32: $(OBJDIR)/b_boehm.o $(OBJDIR)/warmup.o $(OBJDIR)/b_cluster_expansion_32.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/b_boehm_64: $(OBJDIR)/b_boehm.o $(OBJDIR)/warmup.o $(OBJDIR)/b_cluster_expansion_64.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/b_boehm_128: $(OBJDIR)/b_boehm.o $(OBJDIR)/warmup.o $(OBJDIR)/b_cluster_expansion_128.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/b_boehm_256: $(OBJDIR)/b_boehm.o $(OBJDIR)/warmup.o $(OBJDIR)/b_cluster_expansion_256.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/b_boehm_512: $(OBJDIR)/b_boehm.o $(OBJDIR)/warmup.o $(OBJDIR)/b_cluster_expansion_512.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/b_boehm_1024: $(OBJDIR)/b_boehm.o $(OBJDIR)/warmup.o $(OBJDIR)/b_cluster_expansion_1024.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/c_boehm: $(OBJDIR)/c_boehm.o $(OBJDIR)/warmup.o $(OBJDIR)/c_cluster_expansion.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/deviceQuery.o: nvidia/deviceQuery.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

.PHONY.: clean
clean:
	rm -rf $(OBJDIR)
