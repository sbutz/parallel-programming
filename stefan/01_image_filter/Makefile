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

$(OBJDIR)/%.o: %.cu jpeg.h util.h
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJDIR)/filter_bw: $(OBJDIR)/filter_bw.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/blur_bw: $(OBJDIR)/blur_bw.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

run: $(OBJDIR)/filter_bw $(OBJDIR)/blur_bw image.jpg
	$(OBJDIR)/filter_bw image.jpg $(OBJDIR)/image_sw.jpg
	$(OBJDIR)/blur_bw $(OBJDIR)/image_sw.jpg $(OBJDIR)/blur_bw.jpg

.PHONY.: clean
clean:
	rm -rf $(OBJDIR)