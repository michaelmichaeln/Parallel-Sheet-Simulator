# ==============================================================================
# Makefile — Parallel Sheet Simulator
# Ankita Kundu & Michael Nguyen — 15-418/618 Final Project
#
# Targets:
#   make seq        Build sequential CPU baseline
#   make v1         Build V1 (naive CUDA, AoS)
#   make v2         Build V2 (SoA layout)
#   make v3         Build V3 (shared memory tiling)
#   make v4         Build V4 (fully optimized)
#   make all        Build everything
#   make bench      Build all + run full benchmark suite
#   make clean      Remove binaries
# ==============================================================================

CXX      := g++
NVCC     := nvcc
CXXFLAGS := -O2 -std=c++17 -march=native
NVFLAGS  := -O2 -std=c++17 -arch=sm_75

BINDIR   := bin
SIMDIR   := simCode

.PHONY: all seq v1 v2 v3 v4 bench clean dirs

all: dirs seq v1 v2 v3 v4

dirs:
	@mkdir -p $(BINDIR) results/outputs results/metrics results/plots results/journal

seq: dirs
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/cloth_sim_seq cloth_sim_seq.cpp

v1: dirs
	$(NVCC) $(NVFLAGS) -o $(BINDIR)/cloth_sim_v1 $(SIMDIR)/cloth_sim_v1.cu

v2: dirs
	$(NVCC) $(NVFLAGS) -o $(BINDIR)/cloth_sim_v2 $(SIMDIR)/cloth_sim_v2.cu

v3: dirs
	$(NVCC) $(NVFLAGS) -o $(BINDIR)/cloth_sim_v3 $(SIMDIR)/cloth_sim_v3.cu

v4: dirs
	$(NVCC) $(NVFLAGS) -o $(BINDIR)/cloth_sim_v4 $(SIMDIR)/cloth_sim_v4.cu

bench: all
	bash scripts/benchmark.sh
	python3 scripts/plot_results.py

clean:
	rm -rf $(BINDIR)
