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

# Grayscale filter
$(OBJDIR)/grayscale: $(OBJDIR)/grayscale.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

$(OBJDIR)/image_sw.jpg: $(OBJDIR)/grayscale image.jpg
	$(OBJDIR)/grayscale image.jpg $@

# Blur
$(OBJDIR)/blur: $(OBJDIR)/blur.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

MARGIN=8
$(OBJDIR)/blur_bw.jpg: $(OBJDIR)/blur $(OBJDIR)/image_sw.jpg
	$(OBJDIR)/blur $(MARGIN) $(OBJDIR)/image_sw.jpg $@

$(OBJDIR)/blur_rgb.jpg: $(OBJDIR)/blur image.jpg
	$(OBJDIR)/blur $(MARGIN) image.jpg $@


all: $(OBJDIR)/image_sw.jpg $(OBJDIR)/blur_bw.jpg $(OBJDIR)/blur_rgb.jpg

.PHONY.: clean
clean:
	rm -rf $(OBJDIR)