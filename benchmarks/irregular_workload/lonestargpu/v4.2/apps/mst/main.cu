/** Minimum spanning tree -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @Description
 * Computes minimum spanning tree of a graph using Boruvka's algorithm.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 * @author Sreepathi Pai <sreepai@ices.utexas.edu>
 */

#include "lonestargpu.h"
#include "gbar.cuh"
#include "cuda_launch_config.hpp"
#include "devel.h"

unsigned findmax(unsigned* input, int size){
    unsigned maxm = input[0];
    for(int i=1;i<size;i++){
        if(input[i] > maxm) maxm = input[i];
    }
    return maxm;
}

__global__ void dinit(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
		eleminwts[id] = MYINFINITY;
		minwtcomponent[id] = MYINFINITY;	
		goaheadnodeofcomponent[id] = graph.nnodes;
		phores[id] = 0;
		partners[id] = id;
		processinnextiteration[id] = false;
	}
}
__global__ void dfindelemin(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
		// if I have a cross-component edge,
		// 	find my minimum wt cross-component edge,
		//	inform my boss about this edge e (atomicMin).
		unsigned src = id;
		unsigned srcboss = cs.find(src);
		unsigned dstboss = graph.nnodes;
		foru minwt = MYINFINITY;
		unsigned degree = graph.getOutDegree(src);
		for (unsigned ii = 0; ii < degree; ++ii) {
			foru wt = graph.getWeight(src, ii);
			if (wt < minwt) {
				unsigned dst = graph.getDestination(src, ii);
				unsigned tempdstboss = cs.find(dst);
				if (srcboss != tempdstboss) {	// cross-component edge.
					minwt = wt;
					dstboss = tempdstboss;
				}
			}
		}
		dprintf("\tminwt[%d] = %d\n", id, minwt);
		eleminwts[id] = minwt;
		partners[id] = dstboss;

		if (minwt < minwtcomponent[srcboss] && srcboss != dstboss) {
			// inform boss.
			foru oldminwt = atomicMin(&minwtcomponent[srcboss], minwt);
			// if (oldminwt > minwt && minwtcomponent[srcboss] == minwt)
			//   {			    
			
			// 	goaheadnodeofcomponent[srcboss],id);	// threads with same wt edge will race.
			// 	dprintf("\tpartner[%d(%d)] = %d init, eleminwts[id]=%d\n", id, srcboss, dstboss, eleminwts[id]);
			//   }
		}
	}
}

__global__ void dfindelemin2(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (id < graph.nnodes) {
		unsigned src = id;
		unsigned srcboss = cs.find(src);

		if(eleminwts[id] == minwtcomponent[srcboss] && srcboss != partners[id] && partners[id] != graph.nnodes)
		  {
		    unsigned degree = graph.getOutDegree(src);
		    for (unsigned ii = 0; ii < degree; ++ii) {
		      foru wt = graph.getWeight(src, ii);
		      if (wt == eleminwts[id]) {
			unsigned dst = graph.getDestination(src, ii);
			unsigned tempdstboss = cs.find(dst);
			if (tempdstboss == partners[id]) {	// cross-component edge.
			  //atomicMin(&goaheadnodeofcomponent[srcboss], id);
			  
			  if(atomicCAS(&goaheadnodeofcomponent[srcboss], graph.nnodes, id) == graph.nnodes)
			    {
			      //printf("%d: adding %d\n", id, eleminwts[id]);
			      //atomicAdd(wt2, eleminwts[id]);
			    }
			}
		      }
		    }
		  }
	}
}



__global__ void verify_min_elem(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
	  if(cs.isBoss(id))
	    {
	      if(goaheadnodeofcomponent[id] == graph.nnodes)
		{
		  //printf("h?\n");
		  return;
		}

	      unsigned minwt_node = goaheadnodeofcomponent[id];

	      unsigned degree = graph.getOutDegree(minwt_node);
	      foru minwt = minwtcomponent[id];

	      if(minwt == MYINFINITY)
		return;
		
	      bool minwt_found = false;
	      //printf("%d: looking at %d def %d minwt %d\n", id, minwt_node, degree, minwt);
	      for (unsigned ii = 0; ii < degree; ++ii) {
		foru wt = graph.getWeight(minwt_node, ii);
		//printf("%d: looking at %d edge %d wt %d (%d)\n", id, minwt_node, ii, wt, minwt);

		if (wt == minwt) {
		  minwt_found = true;
		  unsigned dst = graph.getDestination(minwt_node, ii);
		  unsigned tempdstboss = cs.find(dst);
		  if(tempdstboss == partners[minwt_node] && tempdstboss != id)
		    {
		      processinnextiteration[minwt_node] = true;
		      //printf("%d okay!\n", id);
		      return;
		    }
		}
	      }

	      printf("component %d is wrong %d\n", id, minwt_found);
	    }
	}
}

__global__ void elim_dups(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
	  if(processinnextiteration[id])
	    {
	      unsigned srcc = cs.find(id);
	      unsigned dstc = partners[id];
	      
	      if(minwtcomponent[dstc] == eleminwts[id])
		{
		  if(id < goaheadnodeofcomponent[dstc])
		    {
		      processinnextiteration[id] = false;
		      //printf("duplicate!\n");
		    }
		}
	    }
	}
}

__global__ void dfindcompmin(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
	  if(partners[id] == graph.nnodes)
	    return;

	  unsigned srcboss = cs.find(id);
	  unsigned dstboss = cs.find(partners[id]);
	  if (id != partners[id] && srcboss != dstboss && eleminwts[id] != MYINFINITY && minwtcomponent[srcboss] == eleminwts[id] && dstboss != id && goaheadnodeofcomponent[srcboss] == id) {	// my edge is min outgoing-component edge.
	    if(!processinnextiteration[id]);
	      //printf("whoa!\n");
	    //= true;
	  }
	  else
	    {
	      if(processinnextiteration[id]);
		//printf("whoa2!\n");
	    }
	}
}

__global__ void dfindup(unsigned *mstwt, Graph graph, ComponentSpace csw, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid, bool *repeat, unsigned *count, unsigned* up, unsigned* id){
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nthreads = blockDim.x * gridDim.x;
	up[tid] = (graph.nnodes + nthreads - 1) / nthreads * nthreads;
        id[tid] = tid;
}

__global__ void dfindcompmintwo1(unsigned *mstwt, Graph graph, ComponentSpace csw, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid, bool *repeat, unsigned *count, unsigned* up, unsigned * id, unsigned* srcboss, unsigned* dstboss) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nthreads = blockDim.x * gridDim.x;
        unsigned myid = id[tid];
        if(myid < up[tid]){
	  if(myid < graph.nnodes && processinnextiteration[myid])
	    {
	      srcboss[tid] = csw.find(myid);
	      dstboss[tid] = csw.find(partners[myid]);
	    }
        }
}

__global__ void dfindcompmintwo2(unsigned *mstwt, Graph graph, ComponentSpace csw, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid, bool *repeat, unsigned *count, unsigned* up, unsigned * id, unsigned* srcboss, unsigned* dstboss) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nthreads = blockDim.x * gridDim.x;
        unsigned myid = id[tid];
        if(myid < up[tid]){
	  if (myid < graph.nnodes && processinnextiteration[myid] && srcboss[tid] != dstboss[tid]) {
	    dprintf("trying unify id=%d (%d -> %d)\n", myid, srcboss[tid], dstboss[tid]);
	    if (csw.unify(srcboss[tid], dstboss[tid])) {
	      atomicAdd(mstwt, eleminwts[myid]);
	      atomicAdd(count, 1);
	      dprintf("u %d -> %d (%d)\n", srcboss[tid], dstboss[tid], eleminwts[myid]);
	      processinnextiteration[myid] = false;
	      eleminwts[myid] = MYINFINITY;	// mark end of processing to avoid getting repeated.
	    }
	    else {
	      *repeat = true;
	    }

	    dprintf("\tcomp[%d] = %d.\n", srcboss[tid], csw.find(srcboss[tid]));
	  }
          myid += nthreads;
          up[tid] = myid;
	}
}

int main(int argc, char *argv[]) {
  unsigned *mstwt, hmstwt = 0;
  int iteration = 0;
  Graph hgraph, graph;
  KernelConfig kconf;

  unsigned *partners, *phores;
  foru *eleminwts, *minwtcomponent;
  bool *processinnextiteration;
  unsigned *goaheadnodeofcomponent;
  const int nSM = kconf.getNumberOfSMs();

  double starttime, endtime;
  const size_t compmintwo_res = maximum_residency(dfindcompmintwo1, 384, 0);

  if (argc != 2) {
    printf("Usage: %s <graph>\n", argv[0]);
    exit(1);
  }

  hgraph.read(argv[1]);
  hgraph.cudaCopy(graph);
  //graph.print();

  kconf.setProblemSize(graph.nnodes);
  ComponentSpace cs(graph.nnodes);

  if (cudaMalloc((void **)&mstwt, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating mstwt failed");
  CUDA_SAFE_CALL(cudaMemcpy(mstwt, &hmstwt, sizeof(hmstwt), cudaMemcpyHostToDevice));	// mstwt = 0.

  if (cudaMalloc((void **)&eleminwts, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating eleminwts failed");
  if (cudaMalloc((void **)&minwtcomponent, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating minwtcomponent failed");
  if (cudaMalloc((void **)&partners, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating partners failed");
  if (cudaMalloc((void **)&phores, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating phores failed");
  if (cudaMalloc((void **)&processinnextiteration, graph.nnodes * sizeof(bool)) != cudaSuccess) CudaTest("allocating processinnextiteration failed");
  if (cudaMalloc((void **)&goaheadnodeofcomponent, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating goaheadnodeofcomponent failed");

  kconf.setMaxThreadsPerBlock();

  unsigned prevncomponents, currncomponents = graph.nnodes;

  bool repeat = false, *grepeat;
  CUDA_SAFE_CALL(cudaMalloc(&grepeat, sizeof(bool) * 1));
  CUDA_SAFE_CALL(cudaMemcpy(grepeat, &repeat, sizeof(bool) * 1, cudaMemcpyHostToDevice));

  unsigned edgecount = 0, *gedgecount;
  CUDA_SAFE_CALL(cudaMalloc(&gedgecount, sizeof(unsigned) * 1));
  CUDA_SAFE_CALL(cudaMemcpy(gedgecount, &edgecount, sizeof(unsigned) * 1, cudaMemcpyHostToDevice));

  printf("finding mst.\n");
  starttime = rtclock();

  do {
    ++iteration;
    prevncomponents = currncomponents;
    dinit 		<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    //printf("0 %d\n", cs.numberOfComponentsHost());
    CudaTest("dinit failed");
    dfindelemin 	<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    dfindelemin2 	<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    verify_min_elem 	<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    CudaTest("dfindelemin failed");
    if(debug) print_comp_mins(cs, graph, minwtcomponent, goaheadnodeofcomponent, partners, processinnextiteration);

    do {
      repeat = false;

      CUDA_SAFE_CALL(cudaMemcpy(grepeat, &repeat, sizeof(bool) * 1, cudaMemcpyHostToDevice));
      unsigned* up = (unsigned*)malloc(sizeof(unsigned)* nSM * compmintwo_res * 384);
      unsigned* dev_up, *dev_id, *dev_srcboss,*dev_dstboss;
      CUDA_SAFE_CALL(cudaMalloc(&dev_up, sizeof(unsigned) * nSM * compmintwo_res * 384));
      CUDA_SAFE_CALL(cudaMalloc(&dev_id, sizeof(unsigned) * nSM * compmintwo_res * 384));
      CUDA_SAFE_CALL(cudaMalloc(&dev_srcboss, sizeof(unsigned) * nSM * compmintwo_res * 384));
      CUDA_SAFE_CALL(cudaMalloc(&dev_dstboss, sizeof(unsigned) * nSM * compmintwo_res * 384));
      dfindup<<<nSM * compmintwo_res, 384>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes, grepeat, gedgecount, dev_up, dev_id);
      CUDA_SAFE_CALL(cudaMemcpy(up, dev_up,sizeof(unsigned) * nSM * compmintwo_res * 384, cudaMemcpyDeviceToHost));
      unsigned maxup = findmax(up,nSM * compmintwo_res * 384 );
      for(int pp = 0;pp < maxup;pp++){
        dfindcompmintwo1 <<<nSM * compmintwo_res, 384>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes, grepeat, gedgecount, dev_up, dev_id, dev_srcboss, dev_dstboss);
        CudaTest("dfindcompmintwo1 failed");
        dfindcompmintwo2 <<<nSM * compmintwo_res, 384>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes, grepeat, gedgecount, dev_up, dev_id, dev_srcboss, dev_dstboss);
        CudaTest("dfindcompmintwo2 failed");
      }
		  
      CUDA_SAFE_CALL(cudaMemcpy(&repeat, grepeat, sizeof(bool) * 1, cudaMemcpyDeviceToHost));

      free(up);
      cudaFree(dev_up);
      cudaFree(dev_id);
      cudaFree(dev_srcboss);
      cudaFree(dev_dstboss);
    } while (repeat); // only required for quicker convergence?

    currncomponents = cs.numberOfComponentsHost();
    CUDA_SAFE_CALL(cudaMemcpy(&hmstwt, mstwt, sizeof(hmstwt), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&edgecount, gedgecount, sizeof(unsigned) * 1, cudaMemcpyDeviceToHost));
    printf("\titeration %d, number of components = %d (%d), mstwt = %u mstedges = %u\n", iteration, currncomponents, prevncomponents, hmstwt, edgecount);
  } while (currncomponents != prevncomponents);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  endtime = rtclock();
	
  printf("\tmstwt = %u, iterations = %d.\n", hmstwt, iteration);
  printf("\t%s result: weight: %u, components: %u, edges: %u\n", argv[1], hmstwt, currncomponents, edgecount);
  printf("\truntime [mst] = %f ms.\n", 1000 * (endtime - starttime));

  // cleanup left to the OS.

  return 0;
}
