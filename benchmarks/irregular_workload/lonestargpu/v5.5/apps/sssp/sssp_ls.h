#pragma once
#define SSSP_VARIANT "lonestar"
#include "cutil_subset.h"

////////////////////////////////
// caogao
#ifndef iteration_profiling
#define iteration_profiling
#endif
////////////////////////////////
__global__
void initialize(foru *dist, unsigned int nv) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if (ii < nv) {
		dist[ii] = MYINFINITY;
	}
}

__device__
bool processedge(foru *dist, Graph &graph, unsigned src, unsigned ii, unsigned &dst) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	dst = graph.getDestination(src, ii);
	if (dst >= graph.nnodes) return 0;

	foru wt = graph.getWeight(src, ii);
	if (wt >= MYINFINITY) return 0;

	foru altdist = dist[src] + wt;
	if (altdist < dist[dst]) {
	 	foru olddist = atomicMin(&dist[dst], altdist);
		if (altdist < olddist) {
			return true;
		} 
		// someone else updated distance to a lower value.
	}
	return false;
}
__device__
bool processnode(foru *dist, Graph &graph, unsigned work) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn = work;
	if (nn >= graph.nnodes) return 0;
	bool changed = false;
	
	unsigned neighborsize = graph.getOutDegree(nn);
#ifdef iteration_profiling
  iteration[blockIdx.x][MAXBLOCKS] = neighborsize; 
  unsigned start_time = 0, end_time = 0;
  if (threadIdx.x == 0) { // first thread in block
    start_time = clock();
  }
#endif
	for (unsigned ii = 0; ii < neighborsize; ++ii) {
		unsigned dst = graph.nnodes;
		for olddist = processedge(dist, graph, nn, ii, dst);
		if (olddist) {
			canged = true;
		}
	}
#ifdef iteration_profiling
  if (threadIdx.x == 0) { // first thread in block
    end_time = clock();
    run_time[blockIdx.x][MAXBLOCKS] = end_time - start_time; 
  }
#endif
	return changed;
}

__global__
#ifdef thread_profiling
void drelax(foru *dist, Graph graph, bool *changed, code_block **code_blocks) {
#else
void drelax(foru *dist, Graph graph, bool *changed) {
#endif
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned start = id * (MAXBLOCKSIZE / blockDim.x), end = (id + 1) * (MAXBLOCKSIZE / blockDim.x); // caogao: essentially (end - start) = MAXBLOCKSIZE / blockDim.x = 1

/*#ifdef thread_profiling
  code_blocks[id][0].iteration = (end - start); 
  code_blocks[id][0].parallel = true;
#endif*/
#ifdef iteration_profiling
  __shared__ int iteration[blockDim.x][MAXBLOCKS]; 
  __shared__ unsigned run_time[blockDim.x][MAXBLOCKS]; 
#endif
 	for (unsigned ii = start; ii < end; ++ii) {
		if (processnode(dist, graph, ii)) {
			*changed = true;
		}
	}
#ifdef iteration_profiling
  synchthread();
  if (threadIdx.x == 0) { // first thread in block
    for (i = 0; i < blockDim.x; i++)
      for (j = 0; j < MAXBLOCKS; i++)
        printf("iteration%d\ttime%u\t", iteration[i][j], run_time[i][j]);
  }
#endif
/*#ifdef thread_profiling
  code_blocks[id][0].runtime = stop_time - start_time;
  code_blocks[id][0].starttime = start_time;
#endif*/
}


void sssp(foru *hdist, foru *dist, Graph &graph, long unsigned totalcommu)
{
	foru foruzero = 0.0;
	bool *changed, hchanged;
	int iteration = 0;
	double starttime, endtime;
	KernelConfig kconf;
  
	kconf.setProblemSize(graph.nnodes);
	cudaMemcpy(&dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");

#ifdef thread_profiling
#define MAX_KERNELS = 50;       // essentially number of iterations for this program
#define MAX_CODE_BLOCKS = 1;    // only one code block for this kernel
#define MAX_ITERATIONS = 50;    // max iterations in kernel
  void print_profile(const char *filename, code_block ***code_blocks); 
  unsigned *iteration_time, **hiteration_time;
  code_block *code_blocks, ***hcode_blocks; 
  // [ [ [which code block] which thread] which kernel]
	if (cudaMalloc((void **)&code_blocks, MAX_CODE_BLOCKS * kconf.getNumberOfBlocks() * kconf.getNumberOfBlockThreads() * MAX_KERNELS * sizeof(code_block)) != cudaSuccess) CudaTest("allocating code_blocks failed");
	if (cudaMalloc((void **)&iteration_time, MAX_ITERATIONS * MAX_KERNELS * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating code_blocks failed");
#endif

	printf("solving.\n");
	starttime = rtclock();
	do {
		++iteration;
		hchanged = false;

		cudaMemcpy(changed, &hchanged, sizeof(hchanged), cudaMemcpyHostToDevice);

#ifdef iteration_profiling
	printf("TEST\tnew_kernel_iteration\t%d\n", iteration);
	printf("TEST\t#Blocks\t%d\t#Threads_per_block\t%d\n", kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads());
#endif
#ifdef thread_profiling
		drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed, code_blocks[iteration-1]);   // kernel # : [iteration-1] 
#else
		drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed);
#endif
		CudaTest("solving failed");

		CUDA_SAFE_CALL(cudaMemcpy(&hchanged, changed, sizeof(hchanged), cudaMemcpyDeviceToHost));
	} while (hchanged);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock(); // changed from lsg (for now) which included memcopies of graph too.

#ifdef thread_profiling
  print_profile("sssp-profile.txt", code_blocks, hcode_blocks, MAX_CODE_BLOCKS * kconf.getNumberOfBlocks() * kconf.getNumberOfBlockThreads() * MAX_KERNELS * sizeof(code_block));
#endif

	CUDA_SAFE_CALL(cudaMemcpy(hdist, dist, graph.nnodes * sizeof(foru), cudaMemcpyDeviceToHost));
	totalcommu += graph.nnodes * sizeof(foru);
	
	printf("\titerations = %d communication = %.3lf MB.\n", iteration, totalcommu * 1.0 / 1000000);

	printf("\truntime [%s] = %f ms\n", SSSP_VARIANT, 1000 * (endtime - starttime));
}
