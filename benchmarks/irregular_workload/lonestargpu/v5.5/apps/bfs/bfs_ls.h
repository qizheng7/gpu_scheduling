/////////////////////////////////////
/////////////////////////////////////
//// assgin Nodes to head Nodes only in balanced_nodes
//TODO: not using "edges_needed" even after residues are used
//TODO: run balance node only once
//TODO: deal with more than one outer iterations
#define BFS_VARIANT "lonestar"
#include "cutil_subset.h"

///////////////////////////////////////////
//caogao
#define WARP_SIZE 32
#define MAX_WARPS 32
#define MAX_GROUPS 64
#define MAX_BLOCKS 1024
#define MAX_ITER 1024
#define MAX_NODES 5 //max number of nodes for a thread
#include "nodes.h"
#define TIMING 1
//#define DEBUG 1
//#define TEST 1
#define ONE_WARP 1
#define UNBALANCED 1

#ifdef DEBUG
__device__
void printout(head_Node *head_nodes, Node *node_warps, const unsigned tid, const unsigned wid){
  Node* node = head_nodes->next; 
////////////////////////
///debug
  if (node == NULL) {
    printf("break\n");
  } else {
////////////////////
    printf("printout: total_length %u\ttotal_nodes %u\tnode_ptr %u\tnode %u\tstart %u\tlength %utid %u\twid %u\n", 
            head_nodes->total_length, head_nodes->total_nodes, node, node->node, node->start, node->length, tid, wid);
    while(node->next != NULL) {
      node = node->next;
      printf("node %u\tstart %u\tlength %u\n", node->node, node->start, node->length);
    };
  }
}
#endif

__device__
//void next_tid(unsigned *tid, unsigned *residue_left, unsigned *edges_thread) {
void next_tid(unsigned *tid, unsigned *edges_thread) {
  (*tid)++;
//  if (*residue_left > 0) {
//    (*residue_left)--;
//    if (residue_left == 0) (*edges_thread)--;
//  }
}

__device__
bool increment(unsigned *ptr, const int bound) {
  (*ptr)++;
  if (*ptr >= bound) {
    printf("max node numbers reached\n");
    return false;
  } 
  return true;
}

__device__
void sort_node_graph(Node_graph *node_graphs, unsigned *sort) {
  unsigned i, k, swap;
  for (i = 0; i < WARP_SIZE; i++) {
    sort[i] = i;
  }
  for (i = 0; i < WARP_SIZE; i++) 
    printf("before length: %u, id: %u\n", node_graphs[sort[i]].length, sort[i]);
  for (i = 0 ; i < (WARP_SIZE - 1); i++)
  {
    for (k = 0 ; k < (WARP_SIZE - i - 1); k++)
    {
      if (node_graphs[sort[k]].length > node_graphs[sort[k+1]].length) /* For decreasing order use < */
      {
        swap = sort[k];
        sort[k] = sort[k+1];
        sort[k] = swap;
      }
    }
  }  
  for (i = 0; i < WARP_SIZE; i++) 
    printf("after length: %u, id: %u\n", node_graphs[sort[i]].length, sort[i]);
}

__device__
//bool balance_nodes(Node_graph *node_graphs, Node_threads *nodes, 
bool balance_nodes(Node_graph *node_graphs, Node_threads *nodes, 
//                    const unsigned average, const unsigned residue, const unsigned wid, const unsigned bid) {
                    const unsigned average, const unsigned wid, const unsigned bid) {
  unsigned node_thread;  // current node in thread
  unsigned edges_needed;  // edges still needed to reach average for each thread
  unsigned edges_per_thread = average;    //edges for each thread
  bool too_many_nodes = false;
  unsigned runs;
  unsigned start_node = blockIdx.x * blockDim.x + threadIdx.x;  // start node idx 
  unsigned node = 0;      // current node in node_graphs
  unsigned sort[WARP_SIZE];
//  if (bid == 3 && wid == 31)
//  sort_node_graph(node_graphs, sort); 
//  unsigned residue_left = residue;
#ifdef DEBUG
  if (bid == 3 && wid == 31)
  for (unsigned atid = 0; atid < WARP_SIZE; atid++) {
    printf("node_length: %u node_start: %u\n", (node_graphs + atid)->length, (node_graphs + atid)->start);
  }
#endif
  for (unsigned tid = 0; tid < WARP_SIZE; tid++) {
    edges_needed = edges_per_thread;
    node_thread = 0;    //current node in the thread
    runs = 0;
    nodes[tid].total_nodes = 0;
    if (runs > 256) {printf("beginning max runs reached: wid: %u, bid: %u\n", wid, bid); return false;}
    do {
    if (node_graphs[node].length > edges_needed)  { // enough edges; need to break nodes
      // create new node
//      nodes[tid].nodes[node_thread].node = node_graphs[node].node;
      nodes[tid].nodes[node_thread].node = start_node + node;
      nodes[tid].nodes[node_thread].start = (node_graphs + node)->start;
      nodes[tid].nodes[node_thread].length = edges_needed;
      nodes[tid].total_nodes++;
      // remove edges from node_graphs[node]
      (node_graphs + node)->length -= edges_needed;
      (node_graphs + node)->start += edges_needed;
#ifdef DEBUG
    if (bid == 3 && wid == 31)
      printf("break node %u\tedge_needed %u\tnew_length %u\tnew_start %u\ttotal_nodes %u\ttid\t%u\n",
            node, edges_needed, (node_graphs + node)->length, (node_graphs + node)->start,
            nodes[tid].total_nodes, tid
    );
#endif
      break;
    } else {  // not enough edges; grab the entire node 
      // create new node
//      nodes[tid].nodes[node_thread].node = *(node_graphs + node)->node;
      nodes[tid].nodes[node_thread].node = start_node + node;
      nodes[tid].nodes[node_thread].start = (node_graphs + node)->start;
      nodes[tid].nodes[node_thread].length = (node_graphs + node)->length;
      nodes[tid].total_nodes++;
      node_thread++;
      // reduce edges_needed
      edges_needed -= (node_graphs + node)->length;
#ifdef DEBUG
    if (bid == 3 && wid == 31)
      printf("grab node %u\tedge_needed %u\tnew_length %u\tnew_start %u\ttotal_nodes %u\ttid\t%u\n",
            node, edges_needed, (node_graphs + node)->length, (node_graphs + node)->start,
            nodes[tid].total_nodes, tid
    );
#endif
      // move to the next node
      node++;
      if (edges_needed == 0) break;   // thread done
      if (node == WARP_SIZE) break;   // all done
    }
    // if this thread has too many nodes
      if (nodes[tid].total_nodes == MAX_NODES) {
        too_many_nodes = true;
        break;
      }
    // if this thread runs too many times
      runs++;
      if (runs > 256) {printf("max runs reached: wid: %u, bid: %u\n", wid, bid); return false;}
    } while(true);
    if (too_many_nodes) {
      printf("too many nodes: tid %u, wid %u, bid: %u\n", tid, wid, bid);
      too_many_nodes = false;
    }
    if (node == WARP_SIZE) break;   // done
  }
  // check if there is still edges left
  if (node != WARP_SIZE) {
    printf("edges left: wid %u, bid: %u\n", wid, bid);
    return false;
  }
  return true;
}
///////////////////////////////////////////

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
//  unsigned wid = threadIdx.x / WARP_SIZE, tid = threadIdx.x % WARP_SIZE, bid = blockIdx.x;
	dst = graph.getDestination(src, ii);
	if (dst >= graph.nnodes) return 0;

	foru wt = 1;

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
bool processnode(foru *dist, Graph &graph, Node_graph *node_graphs, unsigned work, float *timer) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn = work;
	if (nn >= graph.nnodes) return 0;
	bool changed = false;
	
	unsigned neighborsize = graph.getOutDegree(nn);
/////////////////////////////////////////////
///caogao
// timer
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned wid = threadIdx.x / WARP_SIZE, tid = threadIdx.x % WARP_SIZE, bid = blockIdx.x;
  float start_time, end_time;
  if (bid == 0 && wid == 0 && tid == 0)
	  start_time = clock();
// timer
//  __shared__ unsigned average[MAX_WARPS];                     // average number of edges
//  __shared__ Node_graph node_graphs[MAX_WARPS][WARP_SIZE]; 
  __shared__ Node_threads nodes[MAX_WARPS][WARP_SIZE]; 
  __syncthreads();
  if (bid < MAX_BLOCKS) {
//    node_graphs[wid][tid].node = nn;
	  if (nn != id * (MAXBLOCKSIZE / blockDim.x)) {printf("node_graph.node error\n"); return false;}
    (node_graphs + id)->length = neighborsize;
    (node_graphs + id)->start = 0;
//    nodes[wid][tid].total_nodes = 0;
#ifdef DEBUG
  if (bid == 3 && wid == 31)
    printf("tid\t%u\tneighborsize\t%u\n", tid, (node_graphs + id)->length);
#endif
    __syncthreads();
    if (tid == 0) {
      unsigned average = 0;
      for (int i = 0; i < WARP_SIZE; i++)
        average += (node_graphs + id + i)->length;
      average = (average + WARP_SIZE - 1) / WARP_SIZE;
//      residue[wid] = total % WARP_SIZE;
#ifdef ONE_WARP
  if (bid == 3 && 
     (wid == 31 || wid == 32) )
/*#else
  #ifdef UNBALANCED
    if (false)
  #else
    if (true)
  #endif*/
#endif
      balance_nodes((node_graphs + id - (id % WARP_SIZE)) , nodes[wid], average, wid, bid);
    }
  } else {
    printf("bid %u bigger than MAX_BLOCKS\n", bid);
  }
/////////////////////////////////////////////
//caogao
// timer
  if (bid == 0 && wid == 0 && tid == 0) {
	  end_time = clock();
	  *timer = end_time - start_time; 
#ifdef DEBUG
	  printf("balanced_node runtime = %f cycles\n", (end_time - start_time)); 
#endif
/////////////////////////////////////////////
  }
  __syncthreads();
// timer
#ifdef UNBALANCED
  if (false)
#else
  #ifdef ONE_WARP
    if (bid == 0 &&
       (wid == 0 || wid == 1) )
  #else
    if (true)
  #endif 
#endif
  {
// timer
    if (bid == 0 && wid == 0 && (tid == 0 || tid == 1))
  	  start_time = clock();
// timer
/*
    unsigned node_cnt = head_node[wid][tid].total_nodes; 
    if (node_cnt == 0) return changed;
    Node* node = head_node[wid][tid].next; 
  while (true) {
    for (unsigned ii = node->start; ii < (node->start + node->length); ++ii) {
      unsigned dst = graph.nnodes;
      foru olddist = processedge(dist, graph, node->node, ii, dst);
  #ifdef DEBUG
      if (bid == 8 && wid == 0 && ((tid == 0) || (tid == 1))) 
        printf("balanced: processedge:%u\t%u\t%u\n", bid, wid, tid);
  #endif
      if (olddist) {
        changed = true;
      }
////////////////////
    if (bid == 0 && wid == 0 && (tid == 0 || tid == 1)) {
      end_time = clock();
      printf("node: %u edge: %u node runtime = %f cycles\n", node->node, ii, (end_time - start_time)); 
    }
////////////////////
    }
    node_cnt--;
    if (node_cnt == 0) break;
    node = node->next;
    if (node == NULL) {
      break;
    }
  };
*/
// timer
    if (bid == 0 && wid == 0 && (tid == 0 || tid == 1)) {
      end_time = clock();
	    *(timer + 1) = end_time - start_time; 
//#ifdef DEBUG
      printf("node runtime = %f cycles\n", (end_time - start_time)); 
//#endif
    }
// timer
  } else {
// timer
    if (bid == 0 && wid == 0 && tid == 1) 
	    start_time = clock();
// timer
    for (unsigned ii = 0; ii < neighborsize; ++ii) {
      unsigned dst = graph.nnodes;
      foru olddist = processedge(dist, graph, nn, ii, dst);
      if (olddist) {
        changed = true;
      }
    }
// timer
    if (bid == 0 && wid == 0 && tid == 1) {
      end_time = clock();
	    *(timer + 1) = end_time - start_time; 
//#ifdef DEBUG
      printf("node runtime = %f cycles\n", (end_time - start_time)); 
//#endif
    }
// timer
  }
////////////////////////////
	return changed;
}

__global__
void drelax(foru *dist, Graph graph, bool *changed, Node_graph *node_graphs, float *timer) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned start = id * (MAXBLOCKSIZE / blockDim.x), end = (id + 1) * (MAXBLOCKSIZE / blockDim.x);
  if (id == 0)
    if ((MAXBLOCKSIZE / blockDim.x) > 1) 
      printf("more than 1 outer iterations\n");
// timer
/*  float start_time, end_time;
  unsigned wid = threadIdx.x / WARP_SIZE, tid = threadIdx.x % WARP_SIZE, bid = blockIdx.x;
  if (bid == 0 && wid == 0 && tid == 0)
	  start_time = clock();*/
// timer
	for (unsigned ii = start; ii < end; ++ii) {
		if (processnode(dist, graph, node_graphs, ii, timer)) {
			*changed = true;
		}
	}
// timer
/*  if (bid == 0 && wid == 0 && tid == 0) {
    end_time = clock();
	  *(timer + 1) = end_time - start_time; 
  }*/
// timer
}


void bfs(Graph &graph, foru *dist)
{
	cudaFuncSetCacheConfig(drelax, cudaFuncCachePreferShared);
	foru foruzero = 0.0;
	KernelConfig kconf;
	double starttime, endtime;
	bool *changed, hchanged;
	int iteration = 0;

	kconf.setProblemSize(graph.nnodes);

	kconf.setMaxThreadsPerBlock();
/////////////////////////////////
///caogao
  extern void write_solution(const char *fname, Graph &graph, foru *dist);
	printf("Blocks: %u\n", kconf.getNumberOfBlocks());
  printf("Threads per block: %u\n", kconf.getNumberOfBlockThreads());
///////////////////////////////////////
	printf("initializing.\n");
	assert(kconf.coversProblem(0));
	initialize <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph.nnodes);

	CudaTest("initializing failed");

	cudaMemcpy(&dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");

	printf("solving.\n");
	starttime = rtclock();

////////////////////////////
// caogao

//  node_warp: what nodes should each thread work on
//  Node *node_warp; 
//	if (cudaMalloc((void **)&node_warp, MAX_BLOCKS * MAX_WARPS * MAX_GROUPS * sizeof(Node)) != cudaSuccess) CudaTest("allocating changed failed");
//  printf("size:%ld\tstart_address%d\n", MAX_BLOCKS * MAX_WARPS * MAX_GROUPS * sizeof(Node), node_warp);


// node_graph: what are the length and start point of each node
  Node_graph *node_graphs;
//  Node_graph *node_graphs_bak;
	if (cudaMalloc((void **)&node_graphs, MAX_BLOCKS * MAX_WARPS * WARP_SIZE * sizeof(Node_graph)) != cudaSuccess) CudaTest("allocating changed failed");
//	if (cudaMalloc((void **)&node_graphs_bak, MAX_WARPS * WARP_SIZE * sizeof(Node_graph)) != cudaSuccess) CudaTest("allocating changed failed");

//  timer
  float *timer, node_runtimes[MAX_ITER], balanced_nodes_time[MAX_ITER], node_runtime = 0, balanced_node_time = 0;
  float htimer[2];
  double starttime0[MAX_ITER], endtime0[MAX_ITER], runtime0, startime1[MAX_ITER], endtime1[MAX_ITER], runtime1;
//  htimer = (float *) malloc(2);
	if (cudaMalloc((void **)&timer, 2 * sizeof(float)) != cudaSuccess) CudaTest("allocating changed failed");
//  timer

//////////////////////////
	do {
		++iteration;
		hchanged = false;

		CUDA_SAFE_CALL(cudaMemcpy(changed, &hchanged, sizeof(hchanged), cudaMemcpyHostToDevice));

//////////////////////////
// caogao
//		drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed);
    starttime0[iteration-1] = rtclock();
		drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed, node_graphs, timer);
    endtime0[iteration-1] = rtclock();
//timer
//		CUDA_SAFE_CALL(cudaMemcpy(&htimer, timer, sizeof(timer), cudaMemcpyDeviceToHost));
//    balanced_nodes_time[iteration-1] = htimer[0];
//    node_runtimes[iteration-1] = htimer[1];
    balanced_nodes_time[iteration-1] = 0;
    node_runtimes[iteration-1] = 0;
//timer
//////////////////////////
		CudaTest("solving failed");
#ifdef DEBUG
    printf("iteration:%d\n", iteration);
    switch (iteration) {
    case 1:
	    write_solution("bfs-output_1.txt", graph, dist);
      break;
    case 2:
	    write_solution("bfs-output_2.txt", graph, dist);
      break;
    case 3:
	    write_solution("bfs-output_3.txt", graph, dist);
      break;
    case 4:
	    write_solution("bfs-output_4.txt", graph, dist);
      break;
    case 5:
	    write_solution("bfs-output_5.txt", graph, dist);
      break;
    case 6:
	    write_solution("bfs-output_6.txt", graph, dist);
      break;
    case 7:
	    write_solution("bfs-output_7.txt", graph, dist);
      break;
    default:
	    write_solution("bfs-output.txt", graph, dist);
      break;
    }
#endif
////////////////////////////
		CUDA_SAFE_CALL(cudaMemcpy(&hchanged, changed, sizeof(hchanged), cudaMemcpyDeviceToHost));
	} while (hchanged);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();

  for (int i = 0; i < iteration; i++) {
    balanced_node_time += balanced_nodes_time[i];
    node_runtime += node_runtimes[i];
    runtime0 = 1000000 * (endtime0[i] - starttime0[i]);
  }
  balanced_node_time /= iteration;
  node_runtime /= iteration;
  runtime0 /= iteration;
	
	printf("iterations %u\truntime [%s] = %f ms.\n", iteration, BFS_VARIANT, 1000 * (endtime - starttime));
	printf("balance_node %f\t node_runtime %f drelax_runtime %f\n", balanced_node_time, node_runtime, runtime0);

}
