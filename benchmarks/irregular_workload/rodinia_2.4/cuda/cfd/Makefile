include ../../common/make.config

CUDA_SDK_PATH := $(SDK_DIR)

# Determine the correct version of the cutil library
CUTIL_LIB =  -lcutil
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -e $(SDK_DIR)/lib/libcutil_x86_64.a; then echo T; else echo F; fi), T)
        CUTIL_LIB = -lcutil_x86_64
     endif
endif

NVCC_FLAGS = -arch=sm_10 -Xptxas -v


all: euler3d  euler3d_double  pre_euler3d  pre_euler3d_double 

euler3d: euler3d.cu
	nvcc  -O2 -Xptxas -v euler3d.cu -o euler3d -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB) $(NVCC_FLAGS)

euler3d_double: euler3d_double.cu
	nvcc -Xptxas -v -O3 euler3d_double.cu -o euler3d_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB) $(NVCC_FLAGS)


pre_euler3d: pre_euler3d.cu
	nvcc -Xptxas -v -O3 pre_euler3d.cu -o pre_euler3d -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB) $(NVCC_FLAGS)

pre_euler3d_double: pre_euler3d_double.cu
	nvcc -Xptxas -v -O3 pre_euler3d_double.cu -o pre_euler3d_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB) $(NVCC_FLAGS)


clean:
	rm -f euler3d euler3d_double pre_euler3d pre_euler3d_double *.linkinfo
