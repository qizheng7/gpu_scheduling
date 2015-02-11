CC = gcc
CXX = g++
NVCXX = nvcc

GFLAGS = -O3
FLAGS21 = -arch=sm_20 

SRC_11 = GFC_11.cu
SRC_21 = GFC_21.cu

EXE_11 = GFC11
EXE_21 = GFC21

all: $(EXE_11) $(EXE_21)

$(EXE_11): $(SRC_11)
	$(NVCXX) $(GFLAGS) $(SRC_11) -o $(EXE_11)

$(EXE_21): $(SRC_21)
	$(NVCXX) $(GFLAGS) $(FLAGS21) $(SRC_21) -o $(EXE_21)

FPC: fpc.c
	$(CC) -O3 fpc.c -o fpc

clean:
	rm -rf $(EXE_11) $(EXE_21)
