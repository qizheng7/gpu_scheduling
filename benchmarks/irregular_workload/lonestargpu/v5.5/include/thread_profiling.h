#ifdef THREAD_PROFILING_H
#define THREAD_PROFILING_H

typedef struct {
  unsigned runtime;
  bool parallel;
  unsigned iteration;
  unsigned starttime;
} code_block;   
// each kernel has several code_blocks

void print_profile(const char *filename, code_block ***hcode_blocks, code_block ***code_blocks, size_t size){
	CUDA_SAFE_CALL(cudaMemcpy(hcode_blocks, code_blocks, size, cudaMemcpyDeviceToHost));
  
} 
#endif
