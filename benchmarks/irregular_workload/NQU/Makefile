# Please see README.ISPASS-2009 for instructions
CUDA_INSTALL_PATH = /home/caogao/extlib/cuda-4.2/cuda
NVIDIA_COMPUTE_SDK_LOCATION = /home/qizheng/NVIDIA_GPU_Computing_SDK

BINDIR:=$(shell pwd)/bin/
BINSUBDIR=release
SETENV=export BINDIR=$(BINDIR); \
	   export ROOTDIR=$(NVIDIA_COMPUTE_SDK_LOCATION)/C/src/; \
	   export BINSUBDIR=$(BINSUBDIR); \
	   export BOOST_LIB=/usr/lib64; \
	   export BOOST_ROOT=/usr/include; \
	   export BOOST_VER=""; \
	   export OPENMPI_BINDIR=/usr/lib64/mpi/gcc/openmpi/bin/; 
noinline?=0

.PHONY: check_environment
default: check_environment common 
	$(SETENV) make noinline=$(noinline) -f Makefile.nqu

check_environment:
	 @if [ ! -n "$(CUDA_INSTALL_PATH)" ]; then \
		echo "ERROR *** CUDA_INSTALL_PATH not set; please see README.ISPASS-2009"; \
		exit 201; \
	 elif [ ! -d "$(CUDA_INSTALL_PATH)" ]; then \
	 	echo "ERROR *** CUDA_INSTALL_PATH=$(CUDA_INSTALL_PATH) invalid (directory does not exist)."; \
		exit 202; \
	 elif [ ! -n "$(NVIDIA_COMPUTE_SDK_LOCATION)" ]; then \
		echo "ERROR *** NVIDIA_COMPUTE_SDK_LOCATION not set; please see README.ISPASS-2009"; \
		exit 203; \
	 elif [ ! -d "$(NVIDIA_COMPUTE_SDK_LOCATION)" ]; then \
	 	echo "ERROR *** NVIDIA_COMPUTE_SDK_LOCATION=$(NVIDIA_COMPUTE_SDK_LOCATION) invalid (directory does not exist)."; \
		exit 204; \
	 elif [ ! -f "$(NVIDIA_COMPUTE_SDK_LOCATION)/C/lib/libcutil_x86_64.a" -a  ! -f "$(NVIDIA_COMPUTE_SDK_LOCATION)/C/lib/libcutil.a" ]; then \
	 	echo "ERROR *** could not find $(NVIDIA_COMPUTE_SDK_LOCATION)/C/lib/libcutil_x86_64.a (or libcutil.a)"; \
	 	echo "          Build the NVIDIA GPU Computing SDK; please see README.ISPASS-2009"; \
		exit 205; \
	 else \
		NVCC_PATH=`which nvcc`; \
		if [ $$? = 1 ]; then \
			echo ""; \
			echo "ERROR ** nvcc (from CUDA Toolkit) was not found in PATH but required to build the ISPASS 2009 benchmarks."; \
			echo "         Try adding $(CUDA_INSTALL_PATH)/bin/ to your PATH environment variable."; \
			echo "         Please also be sure to read README.ISPASS-2009 if you have not done so."; \
			echo ""; \
			exit 206; \
		else true; \
		fi \
	 fi

common:
	rm -f ../common; ln -s $(NVIDIA_COMPUTE_SDK_LOCATION)/C/common ../common;

#clean: check_environment common
clean: 
	rm -r ../common
	rm ./obj -rf
	rm ./bin -rf

clean-visualizer:
	rm `find -name "gpgpusim_visualizer__*.log.gz"`

all: default
cleanall: clean
