CC = gcc
CXX = g++
NVCXX = nvcc

GFLAGS = -O3 -arch=sm_20 

SRC_11 = TSP_GPU11.cu
SRC_22 = TSP_GPU22.cu

EXE_11 = TSPGPU11
EXE_22 = TSPGPU22

all: $(EXE_11) $(EXE_22) 

$(EXE_11): $(SRC_11)
	$(NVCXX) $(GFLAGS) $(SRC_11) -o $(EXE_11)

$(EXE_22): $(SRC_22)
	$(NVCXX) $(GFLAGS) $(SRC_22) -o $(EXE_22)

clean:
	rm -rf $(EXE_11) $(EXE_22)
