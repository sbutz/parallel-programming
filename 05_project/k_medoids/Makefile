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

NVCC_FLAGS += -dc -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++14
NVCC_LDFLAGS = -ccbin=g++  -lcuda -lcudart

OBJDIR=./build
REPORTDIR=./report
PLOTDIR=./plots

$(OBJDIR)/%.o: %.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJDIR)/%: $(OBJDIR)/%.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $(NVCC_LDFLAGS) $^

KERNELS =	$(OBJDIR)/01_pam \
	$(OBJDIR)/02_pam_no_sqrt \
	$(OBJDIR)/03_pam_2d \
	$(OBJDIR)/04_pam_single_precision \
	$(OBJDIR)/05_pam_intrinsics \
	$(OBJDIR)/06_pam_no_pow \
	$(OBJDIR)/07_pam_shmem \
	$(OBJDIR)/08_pam_small_block \
	$(OBJDIR)/09_pam_end_early \
	$(OBJDIR)/10_pam_matrix
#KERNELS = $(OBJDIR)/09_pam_end_early

$(OBJDIR)/data.csv:
	python3 scripts/datagen.py > $(OBJDIR)/data.csv

DATASETS = $(OBJDIR)/data.csv

all: $(KERNELS) $(DATASETS)

N_CLUSTERS=3
#NVPROF=nsys profile --stats=true --output="./build/report%n"
run: all ./scripts/visualize.py
	@mkdir -p $(PLOTDIR)

	# Only working if sklearn is installed
	#./medoids.py $(N_CLUSTERS) $(OBJDIR)/data.csv $(OBJDIR)/medoids.csv
	#./scripts/cost.py $(OBJDIR)/data.csv $(OBJDIR)/medoids.csv
	#./scripts/visualize.py $(OBJDIR)/data.csv $(OBJDIR)/medoids.csv $(PLOTDIR)/kmedoids-py.png

	for kernel in $(KERNELS); do \
		echo "Running $$kernel"; \
		$(NVPROF) $$kernel $(N_CLUSTERS) $(OBJDIR)/data.csv $(OBJDIR)/medoids.csv; \
		./scripts/cost.py $(OBJDIR)/data.csv $(OBJDIR)/medoids.csv; \
		./scripts/visualize.py $(OBJDIR)/data.csv $(OBJDIR)/medoids.csv $(PLOTDIR)/kmedoids-$$(basename $$kernel).png; \
	done

ncu: all
	@mkdir -p $(REPORTDIR)
	for kernel in $(KERNELS); do \
		echo "Profiling $$kernel"; \
		ncu \
			--config-file off \
			--export $(REPORTDIR)/ncu-$$(basename $$kernel) \
			--force-overwrite \
			--set full \
			--import-source yes  \
			--source-folder . \
			$$kernel $(N_CLUSTERS) $(OBJDIR)/data.csv $(OBJDIR)/medoids.csv; \
	done

$(REPORTDIR)/perf.csv: ./scripts/perf.py
	@mkdir -p $(REPORTDIR)
	python3 ./scripts/perf.py $(REPORTDIR)/perf.csv

plots: $(REPORTDIR)/perf.csv ./scripts/plots.py
	@mkdir -p $(PLOTDIR)
	python3 ./scripts/plots.py $(REPORTDIR)/perf.csv

.PHONY.: clean
clean:
	rm -rf $(OBJDIR)
	rm -rf report*.nsys-rep report*.sqlite