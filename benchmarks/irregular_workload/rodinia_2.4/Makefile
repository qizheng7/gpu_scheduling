include common/make.config

RODINIA_BASE_DIR := $(shell pwd)

CUDA_BIN_DIR := $(RODINIA_BASE_DIR)/bin/linux/cuda

CUDA_DIRS := bfs b+tree

all: CUDA
CUDA: 
	cd cuda/bfs; 			make;	cp bfs $(CUDA_BIN_DIR)
	cd cuda/b+tree;			make;	cp b+tree.out $(CUDA_BIN_DIR)
	
clean: CUDA_clean

CUDA_clean:
	cd $(CUDA_BIN_DIR); rm -f *
	for dir in $(CUDA_DIRS) ; do cd cuda/$$dir ; make clean ; cd ../.. ; done
