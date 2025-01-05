HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

SRCDIR = ./src
OUTDIR = ./bin

CXXFLAGS =-g -std=c++14

CXX=$(HIPCC)

all: $(OUTDIR)/histogram_hip $(OUTDIR)/deviceQuery_hip

$(OUTDIR)/histogram_hip: Makefile $(SRCDIR)/histogram_hip.cpp
	mkdir -p $(OUTDIR)
	$(HIPCC) $(CXXFLAGS) $(SRCDIR)/histogram_hip.cpp -c -o $(OUTDIR)/histogram_hip.o
	$(HIPCC) $(OUTDIR)/histogram_hip.o -o $(OUTDIR)/histogram_hip

$(OUTDIR)/deviceQuery_hip: Makefile nvidia/deviceQuery_hip.cpp
	mkdir -p $(OUTDIR)
	$(HIPCC) $(NVCC_FLAGS) nvidia/deviceQuery_hip.cpp -c -o $(OUTDIR)/deviceQuery_hip.o
	$(HIPCC) $(GENCODE_FLAGS) $(OUTDIR)/deviceQuery_hip.o -o $(OUTDIR)/deviceQuery_hip

.PHONY.: clean
clean:
	rm -f  $(OUTDIR)/*