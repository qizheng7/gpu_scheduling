#pragma once
#define SSSP_VARIANT "lonestar"
#include "cutil_subset.h"

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
	for (unsigned ii = 0; ii < neighborsize; ++ii) {
		unsigned dst = graph.nnodes;
		foru olddist = processedge(dist, graph, nn, ii, dst);
		if (olddist) {
			changed = true;
		}
	}
	return changed;
}

__global__
#ifdef thread_profiling
void drelax(foru *dist, Graph graph, bool *changed, code_block **code_blocks) {
#else
void drelax(foru *dist, Graph graph, bool *changed) {
#endif
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned start = id * (MAXBLOCKSIZE / blockDim.x), end = (id + 1) * (MAXBLOCKSIZE / blockDim.x);

#ifdef thread_profiling
  code_blocks[id][0].iteration = (end - start); 
  code_blocks[id][0].parallel = true;
  unsigned start_time = 0, stop_time = 0;
  start_time = clock();
#endif
 	for (unsigned ii = start; ii < end; ++ii) {
		if (processnode(dist, graph, ii)) {
			*changed = true;
		}
	}
#ifdef thread_profiling
  stop_time = clock();
  code_blocks[id][0].runtime = stop_time - start_time;
  code_blocks[id][0].starttime = start_time;
#endif
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
  void print_profile(const char *filename, code_block ***code_blocks); 
  code_block ***code_blocks, ***hcode_blocks; 
  // [ [ [which code block] which thread] which kernel]
	if (cudaMalloc((void **)&code_blocks, MAX_CODE_BLOCKS * kconf.getNumberOfBlocks() * kconf.getNumberOfBlockThreads() * MAX_KERNELS * sizeof(code_block)) != cudaSuccess) CudaTest("allocating code_blocks failed");
#endif

	printf("solving.\n");
	starttime = rtclock();
	do {
		++iteration;
		hchanged = false;

		cudaMemcpy(changed, &hchanged, sizeof(hchanged), cudaMemcpyHostToDevice);

#ifdef thread_profiling
		drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed, code_blocks[iteration-1]);   // kernel # : [iteration-1] 
#endif
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
