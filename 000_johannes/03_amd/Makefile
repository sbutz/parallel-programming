HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

SRCDIR = ./src
OUTDIR = ./bin

CXXFLAGS =-g -std=c++17

CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS) 
	$(HIPCC) $(OBJECTS) -o $@

$(OUTDIR)/histogram_hip: Makefile $(SRCDIR)/histogram_hip.cpp
	mkdir -p $(OUTDIR)
	$(HIPCC) $(CXXFLAGS) $(SRCDIR)/histogram_hip.cpp -c -o $(OUTDIR)/histogram_hip.o
	$(HIPCC) $(OUTDIR)/histogram_hip.o -o $(OUTDIR)/histogram_hip

.PHONY.: clean
clean:
	rm -f  $(OUTDIR)/*